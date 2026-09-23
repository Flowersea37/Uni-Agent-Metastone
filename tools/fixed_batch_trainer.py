"""Isolated fixed-update-batch trainer used only by the benchmark entrypoint."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import torch
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.utils.gpu_memory_recorder import append_gpu_memory_step


def _tensor_digest(data) -> str:
    digest = hashlib.sha256()
    for key in sorted(data.keys(include_nested=True, leaves_only=True)):
        value = data.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().cpu()
        digest.update(str(key).encode())
        digest.update(str(tensor.dtype).encode())
        if getattr(tensor, "is_nested", False):
            # The V1 transfer queue uses jagged NestedTensor, whose packed
            # values are stable while regular shape/size ops are unsupported.
            tensor = tensor.values()
        else:
            digest.update(str(tuple(tensor.shape)).encode())
        tensor = tensor.contiguous()
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@register_trainer("fixed_batch_colocate_async")
class FixedBatchPPOTrainer(PPOTrainerColocateAsync):
    """Dump or replay the exact TensorDict presented to ``update_actor``.

    This class is registered only when ``tools.fixed_batch_main_ppo`` is used;
    the normal VERL trainer registry and normal training entrypoint are untouched.
    """

    def __init__(self, config):
        super().__init__(config)
        self._fixed_mode = os.environ.get("VERL_FIXED_BATCH_MODE", "").lower()
        self._fixed_dir = Path(os.environ["VERL_FIXED_BATCH_PATH"])
        self._audit_path = Path(os.environ.get("VERL_FIXED_BATCH_AUDIT", f"{self._fixed_dir}.audit.jsonl"))
        self._fixed_update_index = 0
        if self._fixed_mode not in {"dump", "replay"}:
            raise ValueError("VERL_FIXED_BATCH_MODE must be 'dump' or 'replay'")
        if self._fixed_mode == "replay" and not self._fixed_dir.is_dir():
            raise FileNotFoundError(f"Fixed batch directory does not exist: {self._fixed_dir}")

    def _batch_path(self) -> Path:
        return self._fixed_dir / f"update_{self._fixed_update_index:06d}.pt"

    def _audit(self, digest: str, batch, source: str) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "update_index": self._fixed_update_index,
            "mode": self._fixed_mode,
            "source": source,
            "sha256": digest,
            "samples": len(batch.keys),
            "partition_id": batch.partition_id,
        }
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_actor(self, batch, metrics):
        self._fixed_update_index += 1
        fixed_path = self._batch_path()
        if self._fixed_mode == "dump":
            data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)
            data = data.detach().cpu()
            digest = _tensor_digest(data)
            if fixed_path.exists():
                raise FileExistsError(f"Refusing to overwrite fixed batch: {fixed_path}")
            else:
                fixed_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"data": data, "tags": batch.tags, "sha256": digest, "samples": len(batch.keys)},
                    fixed_path,
                )
            self._audit(digest, batch, "live_batch")
        else:
            if not fixed_path.is_file():
                raise FileNotFoundError(f"Missing fixed batch for update {self._fixed_update_index}: {fixed_path}")
            payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
            data = payload["data"]
            digest = _tensor_digest(data)
            if digest != payload["sha256"]:
                raise RuntimeError("Fixed batch SHA256 mismatch; the saved batch is corrupted")
            if len(data) != len(batch.keys):
                raise ValueError(f"Fixed batch has {len(data)} samples, current batch expects {len(batch.keys)}")
            replay_batch = tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=data,
                tags=payload["tags"],
            )
            replay_batch.extra_info.update(batch.extra_info)
            batch = replay_batch
            self._audit(digest, batch, "fixed_batch")

        metrics["fixed_batch/input_sha256_prefix"] = int(digest[:12], 16)
        return super()._update_actor(batch, metrics)

    def fit_fixed_batches(self) -> None:
        """Update the actor directly from saved batches, without rollout.

        Model/worker initialization is intentionally kept identical to the normal
        trainer, but no AgentLoopManager is created and no generation request is
        submitted.  Only the actor update itself is timed and written to CSV.
        """
        if self._fixed_mode != "replay":
            raise ValueError("fit_fixed_batches is only valid in replay mode")

        batch_paths = sorted(self._fixed_dir.glob("update_*.pt"))
        requested_steps = int(self.config.trainer.total_training_steps)
        if len(batch_paths) < requested_steps:
            raise FileNotFoundError(
                f"Requested {requested_steps} direct updates, but only {len(batch_paths)} fixed batches exist"
            )

        # Normal colocate_async training releases rollout weights/KV cache in
        # on_sample_end() before the actor update.  Direct replay never enters
        # that hook, so do the equivalent once up front; otherwise the sleeping
        # actor and a live vLLM replica compete for the same GPU memory.
        self.checkpoint_manager.abort_replicas()
        self.checkpoint_manager.sleep_replicas()

        self.global_steps = 0
        for step, path in enumerate(batch_paths[:requested_steps], start=1):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            samples = int(payload["samples"])
            keys = [f"fixed-direct-{step}-{i}-{uuid.uuid4().hex}" for i in range(samples)]
            batch = KVBatchMeta(
                partition_id="train",
                keys=keys,
                tags=payload["tags"],
            )
            batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            self.global_steps = step
            metrics = {}
            started = time.perf_counter()
            batch = self._update_actor(batch, metrics)
            update_seconds = time.perf_counter() - started

            data = payload["data"]
            input_ids = data.get("input_ids")
            if not isinstance(input_ids, torch.Tensor):
                raise KeyError(f"Fixed batch lacks tensor input_ids: {path}")
            update_tokens = input_ids.values().numel() if input_ids.is_nested else input_ids.numel()
            metrics.update(
                {
                    "timing_s/update_actor": update_seconds,
                    "timing_s/step": update_seconds,
                    "timing_per_token_ms/update_actor": update_seconds * 1000.0 / update_tokens,
                    "perf/total_num_tokens": update_tokens,
                }
            )
            append_gpu_memory_step(metrics, self.config, step)
            tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
            print(
                f"[fixed-direct] step={step} tokens={update_tokens} "
                f"update_actor_s={update_seconds:.6f} sha256={payload['sha256']}",
                flush=True,
            )

        self._shutdown_dump_executor()
