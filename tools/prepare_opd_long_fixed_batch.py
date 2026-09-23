"""Build a reproducible actor-update batch from an existing Uni-Agent trajectory.

This is benchmark data. It uses recorded token IDs, masks, and rollout logprobs;
teacher logprobs and advantages are synthetic. It exercises the OPD k1 actor
loss but does not measure teacher inference or the exact real teacher loss.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from tools.fixed_batch_trainer import _tensor_digest


def jagged(values: torch.Tensor, samples: int) -> torch.Tensor:
    return torch.nested.nested_tensor([values.clone() for _ in range(samples)], layout=torch.jagged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--trajectory-index", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=131072)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--max-response-tokens", type=int, default=122880)
    args = parser.parse_args()
    if args.steps < 1 or args.samples < 1:
        parser.error("--steps and --samples must be positive")

    arrays = np.load(args.trajectory, allow_pickle=False)
    prefix = f"traj{args.trajectory_index}_"
    required = ["prompt_ids", "response_ids", "response_mask", "response_logprobs"]
    missing = [prefix + key for key in required if prefix + key not in arrays]
    if missing:
        parser.error(f"trajectory lacks {missing}")

    prompt = torch.from_numpy(arrays[prefix + "prompt_ids"].astype(np.int64))
    response = torch.from_numpy(arrays[prefix + "response_ids"].astype(np.int64))
    mask = torch.from_numpy(arrays[prefix + "response_mask"].astype(np.int64))
    rollout_logprobs = torch.from_numpy(arrays[prefix + "response_logprobs"].astype(np.float32))
    if not (response.numel() == mask.numel() == rollout_logprobs.numel()):
        raise ValueError("Recorded response IDs, mask, and logprobs have different lengths")
    if prompt.numel() + response.numel() > args.max_total_tokens:
        raise ValueError("Trajectory exceeds --max-total-tokens; choose another trajectory")
    if prompt.numel() > args.max_prompt_tokens or response.numel() > args.max_response_tokens:
        raise ValueError("Trajectory exceeds the configured prompt or response length")
    if mask.sum().item() == 0:
        raise ValueError("Trajectory has no trainable response tokens")

    tokens = torch.cat([prompt, response])
    full_logprobs = torch.cat([torch.zeros(prompt.numel()), rollout_logprobs])
    teacher_logprobs = full_logprobs.unsqueeze(-1).clone()
    teacher_logprobs[prompt.numel() :] -= 0.1
    values = {
        "prompts": prompt,
        "responses": response,
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
        "position_ids": torch.arange(tokens.numel()),
        "loss_mask": mask,
        "response_mask": mask,
        "old_log_probs": rollout_logprobs,
        "rollout_log_probs": rollout_logprobs,
        "log_probs": full_logprobs,
        "teacher_logprobs": teacher_logprobs,
        "advantages": mask.to(torch.float32),
    }
    data = TensorDict(
        {key: jagged(value, args.samples) for key, value in values.items()},
        batch_size=[args.samples],
    )

    digest = _tensor_digest(data)
    payload = {
        "data": data,
        "tags": [{"status": "finished", "is_padding": False} for _ in range(args.samples)],
        "sha256": digest,
        "samples": args.samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        path = args.output_dir / f"update_{step:06d}.pt"
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=False)
            if existing.get("sha256") != digest:
                raise FileExistsError(f"Existing fixed batch differs: {path}")
        else:
            torch.save(payload, path)
    metadata = {
        "trajectory": str(args.trajectory.resolve()),
        "prompt_tokens": prompt.numel(),
        "response_tokens": response.numel(),
        "trainable_response_tokens": int(mask.sum()),
        "total_tokens": tokens.numel(),
        "sha256": digest,
        "steps": args.steps,
        "samples": args.samples,
        "sample_construction": "The same recorded trajectory is repeated for every sample",
        "scope": "Megatron actor OPD k1 update; teacher logprobs are synthetic and teacher inference is excluded",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
