"""Append one actor-update GPU memory record after every completed train step."""

import csv
import os
import time
from pathlib import Path
from typing import Any


_HEADER = [
    "timestamp",
    "step",
    "project_name",
    "experiment_name",
    "precision",
    "fp8",
    "fp8_recipe",
    "fp4",
    "fp4_recipe",
    "rollout_quantization",
    "sequence_parallel",
    "use_remove_padding",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "update_actor_s",
    "gen_s",
    "step_s",
    "total_tokens",
    "response_tokens",
    "update_tokens",
    "rollout_tokens_per_s",
    "update_tokens_per_s",
]


def _plain_number(value: Any) -> int | float | str:
    """Convert numpy/torch scalar-like values without importing either package."""
    if value is None:
        return ""
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return value if isinstance(value, (int, float)) else str(value)


def _tokens_and_speed(seconds: Any, milliseconds_per_token: Any) -> tuple[Any, Any]:
    """Recover the exact token count used by veRL's per-token timing metric."""
    seconds = _plain_number(seconds)
    milliseconds_per_token = _plain_number(milliseconds_per_token)
    if not isinstance(seconds, (int, float)) or not isinstance(milliseconds_per_token, (int, float)):
        return "", ""
    if seconds <= 0 or milliseconds_per_token <= 0:
        return 0, 0
    tokens = seconds * 1000.0 / milliseconds_per_token
    return int(round(tokens)), tokens / seconds


def append_gpu_memory_step(metrics: dict[str, Any], config: Any, step: int) -> None:
    """Append a CSV row when VERL_GPU_MEMORY_LOG is set and actor metrics exist.

    This runs only in the trainer process, so tensor-parallel workers never race
    while writing the shared file. Failures are warnings and do not stop training.
    """
    # Ray tasks/actors do not reliably inherit variables prefixed on the job
    # entrypoint. Prefer the explicit Hydra value and retain the environment
    # variable for backwards compatibility with direct launches.
    trainer_config = getattr(config, "trainer", {})
    configured_path = trainer_config.get("gpu_memory_log", None)
    path_value = configured_path or os.getenv("VERL_GPU_MEMORY_LOG")
    allocated = metrics.get("actor/perf/max_memory_allocated_gb")
    if not path_value or allocated is None:
        return

    try:
        override = config.actor_rollout_ref.actor.megatron.get("override_transformer_config", {})
        fp8 = override.get("fp8", None)
        fp8_recipe = override.get("fp8_recipe", None)
        fp4 = override.get("fp4", None)
        fp4_recipe = override.get("fp4_recipe", None)
        if fp4 not in (None, "", "null", "none"):
            precision = "fp4"
        elif fp8 not in (None, "", "null", "none"):
            precision = "fp8"
        else:
            precision = str(config.actor_rollout_ref.actor.megatron.get("dtype", "bf16"))
        gen_s = metrics.get("timing_s/gen")
        update_actor_s = metrics.get("timing_s/update_actor")
        response_tokens, rollout_tokens_per_s = _tokens_and_speed(
            gen_s, metrics.get("timing_per_token_ms/gen")
        )
        update_tokens, update_tokens_per_s = _tokens_and_speed(
            update_actor_s, metrics.get("timing_per_token_ms/update_actor")
        )
        row = {
            "timestamp": f"{time.time():.6f}",
            "step": step,
            "project_name": config.trainer.project_name,
            "experiment_name": config.trainer.experiment_name,
            "precision": precision,
            "fp8": "" if fp8 is None else fp8,
            "fp8_recipe": "" if fp8_recipe is None else fp8_recipe,
            "fp4": "" if fp4 is None else fp4,
            "fp4_recipe": "" if fp4_recipe is None else fp4_recipe,
            "rollout_quantization": config.actor_rollout_ref.rollout.get("quantization", None) or "none",
            "sequence_parallel": config.actor_rollout_ref.actor.megatron.get("sequence_parallel", True),
            "use_remove_padding": config.actor_rollout_ref.model.get("use_remove_padding", True),
            "peak_allocated_gb": _plain_number(allocated),
            "peak_reserved_gb": _plain_number(metrics.get("actor/perf/max_memory_reserved_gb")),
            "update_actor_s": _plain_number(update_actor_s),
            "gen_s": _plain_number(gen_s),
            "step_s": _plain_number(metrics.get("timing_s/step")),
            "total_tokens": _plain_number(metrics.get("perf/total_num_tokens")),
            "response_tokens": response_tokens,
            "update_tokens": update_tokens,
            "rollout_tokens_per_s": rollout_tokens_per_s,
            "update_tokens_per_s": update_tokens_per_s,
        }

        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=_HEADER)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())
        print(f"[gpu-memory] step {step} appended to {path}", flush=True)
    except Exception as error:
        print(f"[gpu-memory] warning: could not record step {step}: {error}", flush=True)
