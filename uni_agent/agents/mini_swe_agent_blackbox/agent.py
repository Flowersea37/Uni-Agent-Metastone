"""mini-swe-agent adapter for Uni-Agent.

This version keeps the boundary clean:

* mini-swe-agent's ``DefaultAgent`` loop runs in the Uni-Agent agent process.
* The Uni-Agent sandbox remains only the execution environment.
* mini-swe-agent shell actions are forwarded to ``sandbox.exec_shell(...)``.
* Model calls go through mini-swe-agent's ``LitellmModel`` using
  ``config.model.base_url``.

So we do **not** write a runner script into the sandbox, and we do **not**
install mini-swe-agent inside the sandbox.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


def _response_finish_reason(message_or_error: Any) -> str | None:
    """Read the provider finish reason from a mini-swe message/FormatError."""
    messages = getattr(message_or_error, "messages", None)
    if messages:
        payload = messages[0]
    elif isinstance(message_or_error, dict):
        payload = message_or_error
    else:
        return None
    response = payload.get("extra", {}).get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    reason = choices[0].get("finish_reason")
    return str(reason).lower() if reason is not None else None


_DEFAULT_SYSTEM_TEMPLATE = """You are a software engineering agent.
Use bash to inspect and modify the repository.
When you have completed the task, run:
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
followed by a concise summary of the changes.
"""

_DEFAULT_INSTANCE_TEMPLATE = """Solve the following task in {{ cwd }}:

{{ task }}
"""


class MiniSweAgentBlackboxConfig(AgentConfig):
    """Launch params for using mini-swe-agent with a Uni-Agent sandbox."""

    name: str = "mini_swe_agent_blackbox"

    working_dir: str = Field(default="/testbed", description="Repository path inside the sandbox.")
    step_limit: int = Field(default=30, description="mini-swe-agent model-call budget.")
    command_timeout: int = Field(default=60, description="Timeout for each shell command forwarded to the sandbox.")
    agent_timeout: int = Field(default=1800, description="Wallclock timeout for the host-side mini-swe-agent run.")
    cost_limit: float = Field(default=0.0, description="mini-swe-agent cost limit; 0 disables it.")
    max_consecutive_format_errors: int = Field(default=3)
    system_template: str | None = Field(default=None, description="Override mini-swe-agent system prompt.")
    instance_template: str = Field(default=_DEFAULT_INSTANCE_TEMPLATE)
    quiet_litellm: bool = Field(default=True, description="Suppress LiteLLM INFO logs during rollouts.")


@dataclass
class _SandboxEnvironmentConfig:
    cwd: str = "/testbed"
    timeout: int = 60
    env: dict[str, str] = field(default_factory=dict)


class _SandboxEnvironment:
    """mini-swe-agent Environment that delegates command execution to Uni-Agent's sandbox."""

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        config: _SandboxEnvironmentConfig,
    ) -> None:
        self.sandbox = sandbox
        self.loop = loop
        self.config = config
        self.interaction_s = 0.0
        self.interaction_calls = 0
        self.interaction_durations_s: list[float] = []

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        interaction_started = time.perf_counter()
        command = action.get("command", "")
        workdir = cwd or self.config.cwd
        command_timeout = timeout or self.config.timeout
        try:
            result = self._run_coro(
                self.sandbox.exec_shell(
                    command,
                    workdir=workdir,
                    timeout=command_timeout,
                    env=self.config.env or None,
                )
            )
            output = {
                "output": result.stdout + (result.stderr if result.stderr else ""),
                "returncode": result.exit_code,
                "exception_info": "",
            }
            logger.info(
                "mini_swe_agent_blackbox: command finished returncode=%s output_chars=%s",
                result.exit_code,
                len(output["output"]),
            )
        except Exception as exc:
            output = {
                "output": "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        interaction_s = time.perf_counter() - interaction_started
        self.interaction_s += interaction_s
        self.interaction_calls += 1
        self.interaction_durations_s.append(interaction_s)
        self._check_finished(output)
        return output

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {
            "cwd": self.config.cwd,
            "timeout": self.config.timeout,
            **os.environ,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "cwd": self.config.cwd,
                        "timeout": self.config.timeout,
                        "env": self.config.env,
                    },
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def _check_finished(self, output: dict[str, Any]) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if not lines or lines[0].strip() != "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" or output["returncode"] != 0:
            return
        from minisweagent.exceptions import Submitted

        submission = "".join(lines[1:])
        raise Submitted(
            {
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            }
        )


@register_agent("mini_swe_agent_blackbox")
class MiniSweAgentBlackboxAgent(Agent):
    """Run mini-swe-agent's loop on the host while executing actions in the sandbox."""

    config_model = MiniSweAgentBlackboxConfig

    async def run(self, *, sandbox: Sandbox, messages: list[dict[str, Any]]) -> AgentResult:
        cfg: MiniSweAgentBlackboxConfig = self.config  # type: ignore[assignment]
        if not cfg.model.base_url:
            raise ValueError("mini_swe_agent_blackbox: config.model.base_url is not set")

        task = self._extract_task(messages)
        model_config = self._model_config()
        agent_config = self._agent_config(messages)
        env_config = _SandboxEnvironmentConfig(cwd=cfg.working_dir, timeout=cfg.command_timeout)

        loop = asyncio.get_running_loop()
        outcome = await self._run_agent_in_thread(
            sandbox=sandbox,
            loop=loop,
            task=task,
            model_config=model_config,
            agent_config=agent_config,
            env_config=env_config,
            timeout=cfg.agent_timeout,
        )

        diff_started = time.perf_counter()
        diff = await sandbox.exec_shell("git diff", workdir=cfg.working_dir, timeout=60)
        diff_s = time.perf_counter() - diff_started
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
                "timing": {**info.get("timing", {}), "final_git_diff_s": diff_s},
            },
            finished=not bool(info.get("error")),
        )

    def _extract_task(self, messages: list[dict[str, Any]]) -> str:
        task = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        ).strip()
        if not task:
            raise ValueError("mini_swe_agent_blackbox requires a 'user' message")
        return task

    def _model_config(self) -> dict[str, Any]:
        cfg: MiniSweAgentBlackboxConfig = self.config  # type: ignore[assignment]
        model_kwargs: dict[str, Any] = {
            "api_base": cfg.model.base_url.strip() if cfg.model.base_url else cfg.model.base_url,
            "api_key": cfg.model.api_key,
            "custom_llm_provider": "openai",
        }
        if cfg.model.temperature is not None:
            model_kwargs["temperature"] = cfg.model.temperature
        if cfg.model.top_p is not None:
            model_kwargs["top_p"] = cfg.model.top_p
        if cfg.model.max_tokens_per_turn is not None:
            model_kwargs["max_tokens"] = cfg.model.max_tokens_per_turn
        return {
            "model_name": f"openai/{cfg.model.model_name or 'default'}",
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }

    def _agent_config(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        cfg: MiniSweAgentBlackboxConfig = self.config  # type: ignore[assignment]
        return {
            "system_template": self._system_template(messages),
            "instance_template": cfg.instance_template,
            "step_limit": cfg.step_limit,
            "cost_limit": cfg.cost_limit,
            "wall_time_limit_seconds": cfg.agent_timeout,
            "max_consecutive_format_errors": cfg.max_consecutive_format_errors,
        }

    def _system_template(self, messages: list[dict[str, Any]]) -> str:
        cfg: MiniSweAgentBlackboxConfig = self.config  # type: ignore[assignment]
        if cfg.system_template:
            return cfg.system_template
        system_messages = [
            str(message.get("content", "")).strip()
            for message in messages
            if message.get("role") == "system" and str(message.get("content", "")).strip()
        ]
        return "\n\n".join(system_messages) if system_messages else _DEFAULT_SYSTEM_TEMPLATE

    async def _run_agent_in_thread(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        task: str,
        model_config: dict[str, Any],
        agent_config: dict[str, Any],
        env_config: _SandboxEnvironmentConfig,
        timeout: int,
    ) -> dict[str, Any]:
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}

        def target() -> None:
            try:
                result_holder.update(
                    self._run_agent_sync(
                        sandbox=sandbox,
                        loop=loop,
                        task=task,
                        model_config=model_config,
                        agent_config=agent_config,
                        env_config=env_config,
                    )
                )
            except BaseException as exc:
                error_holder["error"] = exc

        thread = threading.Thread(target=target, name="mini-swe-agent", daemon=True)
        start_time = time.monotonic()
        thread.start()
        while thread.is_alive():
            await asyncio.sleep(0.2)
            if timeout > 0 and time.monotonic() - start_time > timeout:
                break

        # ``threading.Thread`` has no public start timestamp; track timeout with join below.
        thread.join(timeout=0)
        if thread.is_alive():
            raise TimeoutError(f"mini_swe_agent_blackbox: agent exceeded timeout={timeout}s")
        if error_holder:
            raise RuntimeError(f"mini_swe_agent_blackbox: {error_holder['error']}") from error_holder["error"]
        return result_holder

    def _run_agent_sync(
        self,
        *,
        sandbox: Sandbox,
        loop: asyncio.AbstractEventLoop,
        task: str,
        model_config: dict[str, Any],
        agent_config: dict[str, Any],
        env_config: _SandboxEnvironmentConfig,
    ) -> dict[str, Any]:
        try:
            self._configure_runtime_logging()
            from minisweagent.agents.default import DefaultAgent
            from minisweagent.models.litellm_model import LitellmModel
        except ImportError as exc:
            raise ImportError(
                "mini_swe_agent_blackbox requires mini-swe-agent in the Uni-Agent host environment. "
                "Install it with: python -m pip install mini-swe-agent"
            ) from exc

        model = LitellmModel(**model_config)
        env = _SandboxEnvironment(sandbox=sandbox, loop=loop, config=env_config)
        agent = _LoggingDefaultAgent(model, env, **agent_config)
        agent_started = time.perf_counter()
        try:
            result = agent.run(task)
            info = agent.serialize().get("info", {})
        except Exception as exc:
            info = {
                "error": f"{type(exc).__name__}: {exc}",
                "partial": agent.serialize().get("info", {}),
            }
            raise
        agent_s = time.perf_counter() - agent_started
        model_rollout_s = float(getattr(agent, "model_rollout_s", 0.0))
        sandbox_interaction_s = float(env.interaction_s)
        info["timing"] = {
            "gen_s": agent_s,
            "model_rollout_s": model_rollout_s,
            "sandbox_interaction_s": sandbox_interaction_s,
            "agent_overhead_s": max(0.0, agent_s - model_rollout_s - sandbox_interaction_s),
            "model_calls": int(getattr(agent, "model_rollout_calls", 0)),
            "sandbox_calls": int(env.interaction_calls),
            "model_call_durations_s": list(getattr(agent, "model_call_durations_s", [])),
            "model_message_durations_s": list(getattr(agent, "model_message_durations_s", [])),
            "sandbox_call_durations_s": list(env.interaction_durations_s),
        }
        return {
            "result": result,
            "messages": agent.messages,
            "info": info,
        }

    def _configure_runtime_logging(self) -> None:
        cfg: MiniSweAgentBlackboxConfig = self.config  # type: ignore[assignment]
        if not cfg.quiet_litellm:
            return
        for name in ("LiteLLM", "litellm", "litellm_model"):
            logging.getLogger(name).setLevel(logging.WARNING)
        try:
            import litellm

            litellm.set_verbose = False
            if hasattr(litellm, "_turn_on_debug"):
                litellm._turn_on_debug = False
        except Exception:
            pass


class _LoggingDefaultAgent:
    """Small logging shim around mini-swe-agent's DefaultAgent.

    Subclassing is done dynamically in ``__new__`` so importing this module does
    not require mini-swe-agent until the agent is actually selected.
    """

    def __new__(cls, *args, **kwargs):
        from minisweagent.agents.default import DefaultAgent

        class LoggingDefaultAgent(DefaultAgent):
            def query(self) -> dict:
                logger.info(
                    "mini_swe_agent_blackbox: model call %s/%s",
                    self.n_calls + 1,
                    self.config.step_limit or "∞",
                )
                model_started = time.perf_counter()
                message = None
                try:
                    message = super().query()
                except Exception as exc:
                    if _response_finish_reason(exc) == "length":
                        from minisweagent.exceptions import InterruptAgentFlow

                        logger.info("mini_swe_agent_blackbox: trajectory token limit reached; stopping agent")
                        raise InterruptAgentFlow(
                            {
                                "role": "exit",
                                "content": "TrajectoryLengthExceeded",
                                "extra": {"exit_status": "TrajectoryLengthExceeded", "submission": ""},
                            }
                        ) from exc
                    logger.warning("mini_swe_agent_blackbox: model call failed: %s: %s", type(exc).__name__, exc)
                    raise
                finally:
                    model_call_s = time.perf_counter() - model_started
                    self.model_rollout_s = getattr(self, "model_rollout_s", 0.0) + model_call_s
                    self.model_rollout_calls = getattr(self, "model_rollout_calls", 0) + 1
                    if not hasattr(self, "model_call_durations_s"):
                        self.model_call_durations_s = []
                    self.model_call_durations_s.append(model_call_s)
                # Only successful calls produce assistant messages. Keep this
                # separate from all attempts so failed calls cannot shift the
                # per-message timing alignment.
                if message is not None:
                    if not hasattr(self, "model_message_durations_s"):
                        self.model_message_durations_s = []
                    self.model_message_durations_s.append(model_call_s)
                if _response_finish_reason(message) == "length":
                    from minisweagent.exceptions import InterruptAgentFlow

                    logger.info("mini_swe_agent_blackbox: trajectory token limit reached; stopping agent")
                    raise InterruptAgentFlow(
                        {
                            "role": "exit",
                            "content": "TrajectoryLengthExceeded",
                            "extra": {"exit_status": "TrajectoryLengthExceeded", "submission": ""},
                        }
                    )
                actions = message.get("extra", {}).get("actions", [])
                if actions:
                    previews = []
                    for action in actions:
                        command = str(action.get("command", "")).replace("\n", "\\n")
                        previews.append(command[:200])
                    logger.info("mini_swe_agent_blackbox: model requested bash: %s", " | ".join(previews))
                else:
                    logger.info("mini_swe_agent_blackbox: model returned no parsed actions")
                return message

        return LoggingDefaultAgent(*args, **kwargs)
