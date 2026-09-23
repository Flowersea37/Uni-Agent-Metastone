"""Run the existing Megatron recipe with fixed actor-update data and no rollout."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    repo = Path(os.environ.get("REPO_ROOT", "/data/xgq/projects/RLs/uni-agent-metastone"))
    script = (repo / "examples/quickstart/training/train_qwen3p5_dense_test.sh").read_text()
    gpu_count = int(os.environ["BENCH_GPUS"])
    gpu_ids = ",".join(str(i) for i in range(gpu_count))

    # Reuse the reference recipe, but make this benchmark incapable of killing
    # processes from a normal training run.
    script = replace_once(script, 'export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"', f'export CUDA_VISIBLE_DEVICES="{gpu_ids}"')
    script = replace_once(script, "ray stop --force\n", "")
    script = replace_once(
        script,
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \\\n  | sort -u \\\n  | xargs -r kill -9\n",
        "",
    )
    script = replace_once(script, "python3 -m verl.trainer.main_ppo", "python3 -m tools.opd_fixed_batch_main_ppo")
    script = replace_once(
        script, "trainer.v1.trainer_mode=${trainer_mode}", "trainer.v1.trainer_mode=fixed_batch_colocate_async"
    )
    script = replace_once(
        script,
        "transfer_queue.enable=True \\\n",
        "transfer_queue.enable=True \\\n"
        "    distillation.enabled=True \\\n"
        "    distillation.distillation_loss.loss_mode=k1 \\\n"
        "    distillation.distillation_loss.use_task_rewards=False \\\n"
        "    distillation.distillation_loss.use_policy_gradient=True \\\n"
        "    ++ray_kwargs.ray_init.runtime_env.env_vars.OPD_TEACHER_URL=http://127.0.0.1:1 \\\n"
        "    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FIXED_BATCH_MODE=replay \\\n"
        "    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FIXED_BATCH_DIRECT='\"true\"' \\\n"
        "    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FIXED_BATCH_PATH=${VERL_FIXED_BATCH_PATH} \\\n"
        "    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_FIXED_BATCH_AUDIT=${VERL_FIXED_BATCH_AUDIT} \\\n",
    )
    script = replace_once(
        script,
        "-- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1",
        '-- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 BENCH_GPUS="${BENCH_GPUS}" '
        'TP="${TP}" CP="${CP}" PP="${PP}" '
        'OPD_TEACHER_URL=http://127.0.0.1:1 VERL_FIXED_BATCH_MODE=replay '
        'VERL_FIXED_BATCH_DIRECT=True VERL_FIXED_BATCH_PATH="${VERL_FIXED_BATCH_PATH}" '
        'VERL_FIXED_BATCH_AUDIT="${VERL_FIXED_BATCH_AUDIT}"',
    )
    if os.environ.get("BENCH_DRY_RUN") == "1":
        subprocess.run(["bash", "-n", "-s"], input=script, text=True, check=True)
        print(f"Rendered benchmark script passed bash -n for {gpu_count} GPUs")
        return
    subprocess.run(["bash", "-s"], input=script, text=True, cwd=repo, env=os.environ, check=True)


if __name__ == "__main__":
    main()
