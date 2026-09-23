"""Tree rollouts for mini-swe-agent.

The first trajectory is sampled normally.  Every completed mini-swe-agent turn
becomes a node containing both its conversation prefix and its repository
state.  Additional trajectories choose a node uniformly at random, restore
that node, and continue sampling from it.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import shlex
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from pydantic import Field

from ..base import AgentResult
from ..mini_swe_agent_blackbox.agent import (
    MiniSweAgentBlackboxAgent,
    MiniSweAgentBlackboxConfig,
    _SandboxEnvironment,
    _SandboxEnvironmentConfig,
)
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


class MiniSweAgentTreeRolloutConfig(MiniSweAgentBlackboxConfig):
    """Configuration for tree rollout sampling."""

    name: str = "mini_swe_agent_treerollout"
    num_branch_rollouts: int = Field(
        default=1,
        ge=0,
        description="Number of extra trajectories sampled from randomly selected turn nodes.",
    )
    random_seed: int | None = Field(default=None, description="Optional deterministic node-selection seed.")


@dataclass
class _TreeNode:
    node_id: int
    parent_id: int | None
    depth: int
    messages: list[dict[str, Any]]
    n_calls: int
    cost: float
    snapshot_prefix: str


@register_agent("mini_swe_agent_treerollout")
class MiniSweAgentTreeRolloutAgent(MiniSweAgentBlackboxAgent):
    """Return one base rollout plus one result for every requested tree expansion."""

    config_model = MiniSweAgentTreeRolloutConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]]) -> list[AgentResult]:
        cfg: MiniSweAgentTreeRolloutConfig = self.config  # type: ignore[assignment]
        if not cfg.model.base_url:
            raise ValueError("mini_swe_agent_treerollout: config.model.base_url is not set")

        task = self._extract_task(messages)
        loop = asyncio.get_running_loop()
        env_config = _SandboxEnvironmentConfig(cwd=cfg.working_dir, timeout=cfg.command_timeout)
        snapshot_root = f"/tmp/uni-agent-tree-{uuid.uuid4().hex}"
        nodes: list[_TreeNode] = []
        results: list[AgentResult] = []
        rng = random.Random(cfg.random_seed)

        try:
            outcome = await self._run_tree_agent_in_thread(
                sandbox=sandbox,
                loop=loop,
                task=task,
                model_config=self._model_config(),
                agent_config=self._agent_config(messages),
                env_config=env_config,
                timeout=cfg.agent_timeout,
                nodes=nodes,
                snapshot_root=snapshot_root,
                parent_id=None,
                initial_node=None,
            )
            results.append(await self._make_result(sandbox, outcome, trajectory_index=0, branch_node=None))

            for trajectory_index in range(1, cfg.num_branch_rollouts + 1):
                if not nodes:
                    logger.warning("tree rollout stopped: the base trajectory produced no completed turns")
                    break
                branch_node = rng.choice(nodes)
                await self._restore(sandbox, cfg.working_dir, branch_node.snapshot_prefix)
                outcome = await self._run_tree_agent_in_thread(
                    sandbox=sandbox,
                    loop=loop,
                    task=task,
                    model_config=self._model_config(),
                    agent_config=self._agent_config(messages),
                    env_config=env_config,
                    timeout=cfg.agent_timeout,
                    nodes=nodes,
                    snapshot_root=snapshot_root,
                    parent_id=branch_node.node_id,
                    initial_node=branch_node,
                )
                results.append(
                    await self._make_result(
                        sandbox,
                        outcome,
                        trajectory_index=trajectory_index,
                        branch_node=branch_node,
                    )
                )
            return results
        finally:
            await sandbox.exec_shell(f"rm -rf {shlex.quote(snapshot_root)}", timeout=60)

    async def _make_result(
        self,
        sandbox: Sandbox,
        outcome: dict[str, Any],
        *,
        trajectory_index: int,
        branch_node: _TreeNode | None,
    ) -> AgentResult:
        cfg: MiniSweAgentTreeRolloutConfig = self.config  # type: ignore[assignment]
        diff = await sandbox.exec_shell("git diff", workdir=cfg.working_dir, timeout=60)
        result = outcome.get("result", {})
        transcript = outcome.get("messages", [])
        info = outcome.get("info", {})
        return AgentResult(
            output={
                "git_diff": diff.stdout,
                "submission": result.get("submission", ""),
                "exit_status": result.get("exit_status", ""),
            },
            transcript=transcript if isinstance(transcript, list) else [],
            info={
                "exit_status": result.get("exit_status"),
                "git_diff_stderr": diff.stderr,
                "mini_swe_agent_info": info,
                "tree": {
                    "trajectory_index": trajectory_index,
                    "branch_node_id": branch_node.node_id if branch_node else None,
                    "branch_depth": branch_node.depth if branch_node else 0,
                },
            },
            finished=not bool(info.get("error")),
        )

    async def _snapshot(self, sandbox: Sandbox, cwd: str, prefix: str) -> None:
        command = (
            f"mkdir -p {shlex.quote(prefix.rsplit('/', 1)[0])} && "
            f"git diff --binary --no-ext-diff HEAD > {shlex.quote(prefix + '.patch')} && "
            f"git ls-files --others --exclude-standard -z | "
            f"tar --null -T - -czf {shlex.quote(prefix + '.untracked.tar.gz')}"
        )
        result = await sandbox.exec_shell(command, workdir=cwd, timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(f"mini_swe_agent_treerollout: failed to snapshot node: {result.stderr}")

    async def _restore(self, sandbox: Sandbox, cwd: str, prefix: str) -> None:
        command = (
            "git reset --hard -q HEAD && git clean -fdq && "
            f"([ ! -s {shlex.quote(prefix + '.patch')} ] || "
            f"git apply --binary {shlex.quote(prefix + '.patch')}) && "
            f"tar -xzf {shlex.quote(prefix + '.untracked.tar.gz')}"
        )
        result = await sandbox.exec_shell(command, workdir=cwd, timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(f"mini_swe_agent_treerollout: failed to restore node: {result.stderr}")

    async def _run_tree_agent_in_thread(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        task: str,
        model_config: dict[str, Any],
        agent_config: dict[str, Any],
        env_config: _SandboxEnvironmentConfig,
        timeout: int,
        nodes: list[_TreeNode],
        snapshot_root: str,
        parent_id: int | None,
        initial_node: _TreeNode | None,
    ) -> dict[str, Any]:
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def target() -> None:
            try:
                result_holder.update(
                    self._run_tree_agent_sync(
                        sandbox=sandbox,
                        loop=loop,
                        task=task,
                        model_config=model_config,
                        agent_config=agent_config,
                        env_config=env_config,
                        nodes=nodes,
                        snapshot_root=snapshot_root,
                        parent_id=parent_id,
                        initial_node=initial_node,
                    )
                )
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=target, name="mini-swe-agent-tree", daemon=True)
        started = time.monotonic()
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.2)
            if timeout > 0 and time.monotonic() - started > timeout:
                break
        thread.join(timeout=0)
        if thread.is_alive():
            raise TimeoutError(f"mini_swe_agent_treerollout: agent exceeded timeout={timeout}s")
        if error_holder:
            raise RuntimeError(f"mini_swe_agent_treerollout: {error_holder['error']}") from error_holder["error"]
        return result_holder

    def _run_tree_agent_sync(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        task: str,
        model_config: dict[str, Any],
        agent_config: dict[str, Any],
        env_config: _SandboxEnvironmentConfig,
        nodes: list[_TreeNode],
        snapshot_root: str,
        parent_id: int | None,
        initial_node: _TreeNode | None,
    ) -> dict[str, Any]:
        try:
            self._configure_runtime_logging()
            from minisweagent.models.litellm_model import LitellmModel
        except ImportError as exc:
            raise ImportError(
                "mini_swe_agent_treerollout requires mini-swe-agent in the Uni-Agent host environment. "
                "Install it with: python -m pip install mini-swe-agent"
            ) from exc

        env = _SandboxEnvironment(sandbox=sandbox, loop=loop, config=env_config)
        current_parent = parent_id

        def on_turn(agent: Any) -> None:
            nonlocal current_parent
            node_id = len(nodes)
            prefix = f"{snapshot_root}/node-{node_id}"
            future = asyncio.run_coroutine_threadsafe(self._snapshot(sandbox, env_config.cwd, prefix), loop)
            future.result()
            prefix_start = len(initial_node.messages) if initial_node else 0
            branch_assistant_turns = sum(
                1 for message in agent.messages[prefix_start:] if message.get("role") == "assistant"
            )
            node = _TreeNode(
                node_id=node_id,
                parent_id=current_parent,
                depth=(initial_node.depth if initial_node else 0) + branch_assistant_turns,
                messages=copy.deepcopy(agent.messages),
                n_calls=agent.n_calls,
                cost=agent.cost,
                snapshot_prefix=prefix,
            )
            nodes.append(node)
            current_parent = node_id

        model = LitellmModel(**model_config)
        agent = _TreeDefaultAgent(model, env, on_turn=on_turn, **agent_config)
        try:
            if initial_node is None:
                result = agent.run(task)
            else:
                result = agent.run_from_node(
                    messages=copy.deepcopy(initial_node.messages),
                    n_calls=initial_node.n_calls,
                    cost=initial_node.cost,
                )
            info = agent.serialize().get("info", {})
        except Exception as exc:
            info = {"error": f"{type(exc).__name__}: {exc}", "partial": agent.serialize().get("info", {})}
            raise
        return {"result": result, "messages": agent.messages, "info": info}


class _TreeDefaultAgent:
    """Lazy DefaultAgent subclass with a turn hook and prefix continuation."""

    def __new__(cls, *args, on_turn: Callable[[Any], None], **kwargs):
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.exceptions import FormatError, InterruptAgentFlow

        class TreeDefaultAgent(DefaultAgent):
            def step(self) -> list[dict]:
                messages = super().step()
                on_turn(self)
                return messages

            def run_from_node(self, *, messages: list[dict], n_calls: int, cost: float) -> dict:
                self.messages = messages
                self.n_calls = n_calls
                self.cost = cost
                while True:
                    try:
                        self.step()
                        self.n_consecutive_format_errors = 0
                    except FormatError as exc:
                        self.cost += exc.messages[0].get("extra", {}).get("cost", 0.0)
                        self.n_consecutive_format_errors += 1
                        if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                            self.add_messages(
                                *exc.messages,
                                {
                                    "role": "exit",
                                    "content": "RepeatedFormatError",
                                    "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                                },
                            )
                        else:
                            self.add_messages(*exc.messages)
                    except InterruptAgentFlow as exc:
                        self.add_messages(*exc.messages)
                    except Exception as exc:
                        self.handle_uncaught_exception(exc)
                        raise
                    finally:
                        self.save(self.config.output_path)
                    if self.messages[-1].get("role") == "exit":
                        break
                return self.messages[-1].get("extra", {})

        return TreeDefaultAgent(*args, **kwargs)
