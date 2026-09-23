"""Benchmark-only V1 entrypoint that verifies the Megatron topology."""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from tools.fixed_batch_main_ppo import FixedBatchTaskRunner
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_megatron_trainer", version_base=None)
def main(config: DictConfig) -> None:
    actor = config.actor_rollout_ref.actor
    ref = config.actor_rollout_ref.ref
    if actor.strategy != "megatron" or ref.strategy != "megatron":
        raise ValueError("Fixed trajectory benchmark requires Megatron actor and ref strategies")
    expected = (int(os.environ["BENCH_GPUS"]), int(os.environ["TP"]), int(os.environ["CP"]), int(os.environ["PP"]))
    actual = (
        int(config.trainer.nnodes * config.trainer.n_gpus_per_node),
        int(actor.megatron.tensor_model_parallel_size),
        int(actor.megatron.context_parallel_size),
        int(actor.megatron.pipeline_model_parallel_size),
    )
    if actual != expected:
        raise ValueError(f"Megatron GPU/TP/CP/PP mismatch: expected {expected}, got {actual}")
    if config.trainer.v1.trainer_mode != "fixed_batch_colocate_async":
        raise ValueError("Benchmark must use the direct fixed-batch trainer")
    if not config.distillation.enabled or config.distillation.distillation_loss.loss_mode != "k1":
        raise ValueError("Fixed trajectory benchmark requires OPD k1 loss")
    worker_env = config.ray_kwargs.ray_init.runtime_env.env_vars
    for name in (
        "VERL_FIXED_BATCH_MODE",
        "VERL_FIXED_BATCH_DIRECT",
        "VERL_FIXED_BATCH_PATH",
        "VERL_FIXED_BATCH_AUDIT",
    ):
        if not isinstance(worker_env.get(name), str) or not worker_env[name]:
            raise ValueError(f"Ray worker runtime_env is missing string variable {name}")
    print(
        f"Verified Megatron actor/ref and OPD k1 loss: "
        f"GPUs={actual[0]} TP={actual[1]} CP={actual[2]} PP={actual[3]}",
        flush=True,
    )
    auto_set_device(config)
    validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
    run_ppo(config, task_runner_class=FixedBatchTaskRunner)


if __name__ == "__main__":
    main()
