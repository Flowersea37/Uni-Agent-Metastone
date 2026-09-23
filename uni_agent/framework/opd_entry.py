"""OPD-enabled adapter for the Uni-Agent framework rollout path.

This lives beside, rather than modifies, the standard adapter so existing
AgentFrameworkRolloutAdapter recipes keep their original behavior.
"""

from __future__ import annotations

from dataclasses import replace

import ray

from uni_agent.framework.entry import build_gateway_manager
from uni_agent.framework.framework import OpenAICompatibleAgentFramework
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.transferqueue_utils import tq
from verl.workers.config import DistillationConfig
from verl.workers.config.model import HFModelConfig


class OPDOpenAICompatibleAgentFramework(OpenAICompatibleAgentFramework):
    """Agent framework that attaches teacher logprobs before each TQ write."""

    @classmethod
    def from_config(
        cls,
        *,
        config,
        gateway_manager,
        teacher_client=None,
        processor=None,
        reward_loop_worker_handles=None,
    ) -> OPDOpenAICompatibleAgentFramework:
        instance = super().from_config(
            config=config,
            gateway_manager=gateway_manager,
            processor=processor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        instance._distillation_enabled = is_distillation_enabled(distillation_config)
        instance._teacher_key = distillation_config.teacher_key
        instance._teacher_manager = None
        if instance._distillation_enabled:
            if teacher_client is None:
                raise ValueError("OPD agent rollout requires teacher_client when distillation is enabled")
            instance._teacher_manager = AsyncTeacherLLMServerManager(
                config=config,
                teacher_client=teacher_client,
            )
        return instance

    async def _write_session_trajectories_to_tq(
        self,
        *,
        uid: str,
        session_index: int,
        trajectories,
        sample_fields: dict[str, object],
        global_steps: int | None,
        partition_id: str,
    ) -> None:
        if self._distillation_enabled and partition_id == "train":
            routing_key = sample_fields.get(self._teacher_key)
            if hasattr(routing_key, "item"):
                routing_key = routing_key.item()

            enriched = []
            for trajectory in trajectories:
                teacher_ids, teacher_logprobs = await self._teacher_manager.compute_teacher_logprobs_single(
                    sequence_ids=trajectory.prompt_ids + trajectory.response_ids,
                    multi_modal_data=trajectory.multi_modal_data,
                    routing_key=routing_key,
                )
                enriched.append(
                    replace(
                        trajectory,
                        extra_fields={
                            **trajectory.extra_fields,
                            "teacher_ids": teacher_ids,
                            "teacher_logprobs": teacher_logprobs,
                        },
                    )
                )
            trajectories = enriched

        await super()._write_session_trajectories_to_tq(
            uid=uid,
            session_index=session_index,
            trajectories=trajectories,
            sample_fields=sample_fields,
            global_steps=global_steps,
            partition_id=partition_id,
        )


@ray.remote
class OPDAgentFrameworkWorker:
    """Ray host for the OPD-aware framework."""

    def __init__(self, *, config, gateway_manager, teacher_client, reward_loop_worker_handles=None) -> None:
        tq.init()
        model_config: HFModelConfig = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        self.framework = OPDOpenAICompatibleAgentFramework.from_config(
            config=config,
            gateway_manager=gateway_manager,
            teacher_client=teacher_client,
            processor=model_config.processor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

    async def generate_sequences(self, prompts) -> None:
        await self.framework.generate_sequences(prompts)


class OPDAgentFrameworkRolloutAdapter:
    """Trainer-facing Uni-Agent adapter that supports an OPD teacher client."""

    def __init__(self) -> None:
        self.framework_worker = None
        self.gateway_manager = None

    @classmethod
    def create(
        cls,
        *,
        config,
        llm_client,
        teacher_client=None,
        reward_loop_worker_handles=None,
        **_,
    ) -> OPDAgentFrameworkRolloutAdapter:
        if teacher_client is None:
            raise ValueError("OPDAgentFrameworkRolloutAdapter requires teacher_client")

        gateway_manager = build_gateway_manager(config=config, llm_client=llm_client)
        framework_worker = OPDAgentFrameworkWorker.remote(
            config=config,
            gateway_manager=gateway_manager,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        instance = cls()
        instance.framework_worker = framework_worker
        instance.gateway_manager = gateway_manager
        return instance

    def generate_sequences(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        self.framework_worker.generate_sequences.remote(prompts)
        return None

    def generate_sequences_and_wait(self, prompts) -> None:
        if self.framework_worker is None:
            raise RuntimeError("framework must be initialized before generate_sequences")
        ray.get(self.framework_worker.generate_sequences.remote(prompts))
        return None
