#!/usr/bin/env python3
"""Convert a Uni-Agent trajectory dump to a human-readable JSON document."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


CHATML_START = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\r?\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Session directory, trajectory.json, or trajectory.npz")
    parser.add_argument("--model-path", default="/data/xgq/models/Qwen/Qwen3.5-9B")
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    return parser.parse_args()


def resolve_files(path: Path) -> tuple[Path, Path]:
    directory = path if path.is_dir() else path.parent
    return directory / "trajectory.json", directory / "trajectory.npz"


def decode(tokenizer: Any, values: np.ndarray) -> str:
    return tokenizer.decode(values.astype(np.int64).tolist(), skip_special_tokens=False)


def parse_chatml(text: str) -> list[dict[str, Any]]:
    """Split decoded ChatML and normalize tool responses into tool messages."""
    starts = list(CHATML_START.finditer(text))
    history: list[dict[str, Any]] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        content = text[match.end() : end]
        content = re.sub(r"<\|im_end\|>\s*$", "", content).strip()
        role = match.group(1)
        original_role = role
        if role == "user" and content.startswith("<tool_response>"):
            role = "tool"
        history.append(
            {
                "index": index,
                "role": role,
                "content": content,
                "source": "model" if role == "assistant" else ("tool" if role == "tool" else "input"),
                **({"original_role": original_role} if role != original_role else {}),
            }
        )
    return history


def scalar_stats(values: np.ndarray | None) -> dict[str, float] | None:
    if values is None or values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return {"mean": float(finite.mean()), "min": float(finite.min()), "max": float(finite.max())}


def attach_message_timings(history: list[dict[str, Any]], timing: dict[str, Any] | None) -> None:
    """Attach exact per-call durations when the trajectory recorded them.

    Older dumps only contain aggregate durations.  Do not fabricate per-message
    values by dividing the aggregate: calls can differ by orders of magnitude.
    """
    timing = timing or {}
    durations_by_role = {
        "assistant": timing.get("model_message_durations_s", timing.get("model_call_durations_s")),
        "tool": timing.get("sandbox_call_durations_s"),
    }
    positions = {"assistant": 0, "tool": 0}
    for message in history:
        role = message["role"]
        if role not in durations_by_role:
            continue
        durations = durations_by_role[role]
        position = positions[role]
        positions[role] += 1
        if isinstance(durations, list) and position < len(durations):
            message["timing"] = {
                "duration_s": float(durations[position]),
                "kind": "model_rollout" if role == "assistant" else "sandbox_interaction",
                "available": True,
            }
        else:
            message["timing"] = {
                "duration_s": None,
                "kind": "model_rollout" if role == "assistant" else "sandbox_interaction",
                "available": False,
                "reason": "per-call timing was not recorded in this trajectory",
            }


def deduplicated_meta(trajectory_meta: dict[str, Any]) -> dict[str, Any]:
    """Keep timing only in metrics.timing in the readable representation."""
    cleaned = copy.deepcopy(trajectory_meta)
    for field in ("reward_info", "reward_extra_info"):
        value = cleaned.get(field)
        if isinstance(value, dict):
            value.pop("timing", None)
    return cleaned


def main() -> None:
    args = parse_args()
    meta_path, arrays_path = resolve_files(args.path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    index = args.trajectory_index
    if index < 0 or index >= len(meta["trajectories"]):
        raise IndexError(f"trajectory index {index} is outside 0..{len(meta['trajectories']) - 1}")

    prefix = f"traj{index}"
    prompt_ids = arrays[f"{prefix}_prompt_ids"]
    response_ids = arrays[f"{prefix}_response_ids"]
    response_mask = arrays[f"{prefix}_response_mask"].astype(bool)
    logprob_key = f"{prefix}_response_logprobs"
    response_logprobs = arrays[logprob_key] if logprob_key in arrays.files else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    history = parse_chatml(decode(tokenizer, prompt_ids) + decode(tokenizer, response_ids))
    trajectory_meta = meta["trajectories"][index]
    reward_extra = trajectory_meta.get("reward_extra_info") or {}
    reward_info = trajectory_meta.get("reward_info") or {}
    timing = reward_extra.get("timing") if isinstance(reward_extra, dict) else None
    if not isinstance(timing, dict) and isinstance(reward_info, dict):
        timing = reward_info.get("timing")
    timing_metrics = {}
    if isinstance(timing, dict):
        gen_s = timing.get("gen_s")
        model_s = timing.get("model_rollout_s")
        timing_metrics = {
            **timing,
            "non_model_s": max(0.0, gen_s - model_s)
            if isinstance(gen_s, (int, float)) and isinstance(model_s, (int, float))
            else None,
            "model_rollout_percent": model_s / gen_s * 100
            if isinstance(gen_s, (int, float)) and gen_s > 0 and isinstance(model_s, (int, float))
            else None,
        }
    attach_message_timings(history, timing if isinstance(timing, dict) else None)
    model_tokens = int(response_mask.sum())
    response_tokens = int(response_ids.size)
    tool_context_tokens = response_tokens - model_tokens
    model_logprobs = response_logprobs[response_mask] if response_logprobs is not None else None
    document = {
        "session_id": meta.get("session_id"),
        "trajectory_index": index,
        "metrics": {
            **deduplicated_meta(trajectory_meta),
            "prompt_tokens": int(prompt_ids.size),
            "response_tokens": response_tokens,
            "total_tokens": int(prompt_ids.size + response_tokens),
            "model_tokens": model_tokens,
            "tool_context_tokens": tool_context_tokens,
            "model_token_ratio": model_tokens / response_tokens if response_tokens else 0.0,
            "message_count": len(history),
            "assistant_message_count": sum(item["role"] == "assistant" for item in history),
            "tool_message_count": sum(item["role"] == "tool" for item in history),
            "model_logprob": scalar_stats(model_logprobs),
            "timing": timing_metrics or None,
        },
        "history": history,
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
