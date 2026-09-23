"""Dedicated VERL entrypoint for the isolated fixed-update-batch benchmark."""

from __future__ import annotations

from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging


@ray.remote
class FixedBatchTaskRunner:
    """TaskRunner that imports the benchmark-only trainer inside the Ray actor."""

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.import_utils import load_class_from_fqn

        fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_cls = load_class_from_fqn(fqn, "AgentLoopManager") if fqn else AgentLoopManagerTQ
        self.agent_loop_manager = manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        configure_verl_logging()
        import transfer_queue as tq
        from tools import fixed_batch_trainer  # noqa: F401
        from verl.trainer.ppo.v1 import get_trainer_cls

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            trainer_cls = get_trainer_cls(config.trainer.v1.trainer_mode)
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            direct_replay = (
                __import__("os").environ.get("VERL_FIXED_BATCH_DIRECT", "").lower()
                in {"1", "true", "yes"}
            )
            if direct_replay:
                self.trainer.fit_fixed_batches()
            else:
                self.init_agent_loop_manager()
                self.trainer.fit(self.agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if not config.trainer.use_v1:
        raise ValueError("The fixed-batch benchmark requires trainer.use_v1=True")
    run_ppo(config, task_runner_class=FixedBatchTaskRunner)


if __name__ == "__main__":
    main()
