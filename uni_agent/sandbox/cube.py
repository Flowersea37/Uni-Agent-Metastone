from __future__ import annotations

import asyncio
import logging
import os
import shlex
from typing import TYPE_CHECKING, Any

from .base import ExecResult, Sandbox
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig


logger = logging.getLogger(__name__)


def _normalize_api_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    url = value.strip().rstrip("/")
    return url if "://" in url else f"http://{url}"


def _stream_text(stream: Any) -> str:
    """Flatten the output stream returned by different e2b SDK versions."""
    if stream is None:
        return ""
    if isinstance(stream, str):
        return stream
    if not isinstance(stream, (list, tuple)):
        stream = [stream]
    chunks: list[str] = []
    for item in stream:
        value = getattr(item, "value", item)
        chunks.append(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))
    return "".join(chunks)


@register_sandbox("cube")
class CubeSandbox(Sandbox):
    """Create a local Cube MicroVM through the e2b-compatible Python SDK."""

    def __init__(
        self,
        *,
        template: str | None = None,
        api_url: str | None = None,
        **create_kwargs: Any,
    ) -> None:
        self.template = template or os.getenv("CUBE_TEMPLATE_ID")
        self.api_url = _normalize_api_url(api_url or os.getenv("E2B_API_URL"))
        self.create_kwargs = dict(create_kwargs)
        self._sandbox: Any = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> CubeSandbox:
        kwargs = dict(config.sandbox_kwargs)
        template = kwargs.pop("template", None)
        api_url = kwargs.pop("api_url", None)
        return cls(template=template, api_url=api_url, **kwargs)

    async def start(self) -> None:
        if self._sandbox is not None:
            return
        if not self.template:
            raise ValueError("CubeSandbox needs a template: set CUBE_TEMPLATE_ID or sandbox_kwargs.template")
        try:
            from e2b_code_interpreter import Sandbox as E2BSandbox
        except ImportError as exc:
            raise ImportError(
                "CubeSandbox requires e2b-code-interpreter; install the Cube-compatible SDK on every worker"
            ) from exc
        create_kwargs = {"template": self.template, **self.create_kwargs}
        if self.api_url is not None:
            create_kwargs["api_url"] = self.api_url
        self._sandbox = await asyncio.to_thread(E2BSandbox.create, **create_kwargs)

    async def stop(self) -> None:
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is None:
            return
        close = getattr(sandbox, "kill", None) or getattr(sandbox, "close", None)
        if close is not None:
            await asyncio.to_thread(close)

    async def is_alive(self) -> bool:
        return self._sandbox is not None

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise RuntimeError("CubeSandbox not started; call start() first")
        return self._sandbox

    async def exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a command while keeping SDK/data-plane failures distinct from exit 127."""
        try:
            return await self._exec(argv, timeout=timeout, workdir=workdir, env=env)
        except Exception:
            logger.exception(
                "Cube command transport failed: argv=%r workdir=%r timeout=%r env_keys=%s",
                argv,
                workdir,
                timeout,
                sorted((env or {}).keys()),
            )
            raise

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        command = shlex.join(argv)
        run_kwargs: dict[str, Any] = {}
        if timeout is not None:
            run_kwargs["timeout"] = timeout
        if workdir is not None:
            run_kwargs["cwd"] = workdir
        if env is not None:
            run_kwargs["envs"] = env
        result = await asyncio.to_thread(self._require_sandbox().commands.run, command, **run_kwargs)
        return ExecResult(
            exit_code=int(result.exit_code),
            stdout=_stream_text(result.stdout),
            stderr=_stream_text(result.stderr),
        )
