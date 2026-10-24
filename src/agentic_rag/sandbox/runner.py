"""Run untrusted, LLM-written Python in a throwaway Docker container.

Threat model: the script is written by a language model from text that may include web snippets or
document content, so treat it as hostile. The container gets no network, a read-only root
filesystem, a small tmpfs, all Linux capabilities dropped, a non-root user, and hard CPU / memory /
process / time limits. The code travels over stdin, so nothing from the host is mounted.

This is process-level isolation (the same kernel as the host), not a VM. It is a sound control for
a local tool, and not enough for multi-tenant hosting (that would want gVisor or Firecracker).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import SandboxError
from agentic_rag.core.logging import get_logger

log = get_logger(__name__)

_DOCKER_DAEMON_ERROR = 125  # `docker run` itself failed (daemon down, bad flag, image missing)
_OOM_KILLED = 137
# Messages the docker CLI itself prints when it cannot start a container. Newer CLIs exit with 1
# (not 125) when the daemon is down, so the exit code alone is not a reliable signal.
_DOCKER_FAILURE_MARKERS = (
    "failed to connect to the docker api",
    "cannot connect to the docker daemon",
    "error during connect",
    "error response from daemon",
    "no such image",
    "unable to find image",
)


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    duration_s: float

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class CodeRunner(Protocol):
    async def check(self) -> None:
        """Raise SandboxError with an actionable message if code cannot be run."""

    async def run(self, code: str) -> SandboxResult: ...


ProcessFactory = Callable[..., Awaitable[Any]]


async def _drain(
    stream: asyncio.StreamReader, limit: int, *, keep: Literal["head", "tail"]
) -> tuple[bytes, bool]:
    """Read to EOF while buffering at most `limit` bytes, so a script that prints in a loop cannot
    exhaust host memory and the pipe never blocks the container. stdout keeps the head (the
    start of the result); stderr keeps the tail (a traceback's last lines name the error)."""
    kept = bytearray()
    truncated = False
    while chunk := await stream.read(65536):
        if keep == "head":
            room = limit - len(kept)
            if room > 0:
                kept += chunk[:room]
            if len(chunk) > max(room, 0):
                truncated = True
        else:
            kept += chunk
            if len(kept) > limit:
                del kept[: len(kept) - limit]
                truncated = True
    return bytes(kept), truncated


class DockerSandbox:
    def __init__(
        self,
        settings: Settings,
        *,
        docker_bin: str = "docker",
        process_factory: ProcessFactory = asyncio.create_subprocess_exec,
    ) -> None:
        self._s = settings
        self._docker = docker_bin
        self._spawn = process_factory

    def build_command(self, container_name: str) -> list[str]:
        s = self._s
        return [
            self._docker,
            "run",
            "--rm",
            "--pull",
            "never",  # a missing image is a setup error to report, not a silent download
            "-i",
            "--name",
            container_name,
            "--network",
            "none",
            "--memory",
            f"{s.sandbox_memory_mb}m",
            "--memory-swap",
            f"{s.sandbox_memory_mb}m",  # equal to --memory: no swap
            "--cpus",
            str(s.sandbox_cpus),
            "--pids-limit",
            str(s.sandbox_pids),
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--workdir",
            "/tmp",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HOME=/tmp",
            s.sandbox_image,
            "python",
            "-I",  # isolated mode: ignore env vars and the working directory on sys.path
            "-",  # read the script from stdin
        ]

    async def _docker_cmd(self, *args: str) -> tuple[int, str]:
        try:
            proc = await self._spawn(
                self._docker,
                *args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError as exc:
            raise SandboxError(
                "The `docker` command was not found. Install Docker Desktop and make sure it is "
                "on PATH, or set CODE_WORKER_ENABLED=false."
            ) from exc
        out, _ = await proc.communicate()
        return proc.returncode if proc.returncode is not None else -1, out.decode(
            "utf-8", errors="replace"
        )

    async def check(self) -> None:
        code, out = await self._docker_cmd("image", "inspect", self._s.sandbox_image)
        if code == 0:
            return
        lowered = out.lower()
        if "no such image" in lowered or "no such object" in lowered:
            raise SandboxError(
                f"Sandbox image {self._s.sandbox_image!r} is not pulled. "
                f"Run: docker pull {self._s.sandbox_image}"
            )
        raise SandboxError(
            f"Docker is not usable (is Docker Desktop running?): {out.strip()[:300]}"
        )

    async def _kill(self, name: str) -> None:
        try:
            await asyncio.wait_for(self._docker_cmd("kill", name), timeout=5)
        except (TimeoutError, SandboxError, OSError):
            log.warning("sandbox_kill_failed", container=name)

    async def run(self, code: str) -> SandboxResult:
        s = self._s
        name = f"agentic-rag-sbx-{uuid.uuid4().hex[:12]}"
        limit = s.sandbox_max_output_chars * 4  # bytes; decoded and clipped below
        started = time.perf_counter()
        try:
            proc = await self._spawn(
                *self.build_command(name),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SandboxError("The `docker` command was not found.") from exc

        async def feed() -> None:
            try:
                proc.stdin.write(code.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the container exited before reading everything; its exit code tells why
            finally:
                with contextlib.suppress(OSError):
                    proc.stdin.close()

        async def collect() -> tuple[bytes, bool, bytes, bool, int]:
            (out, out_trunc), (err, err_trunc), _ = await asyncio.gather(
                _drain(proc.stdout, limit, keep="head"),
                _drain(proc.stderr, limit, keep="tail"),
                feed(),
            )
            return out, out_trunc, err, err_trunc, await proc.wait()

        timed_out = False
        try:
            out, out_trunc, err, err_trunc, exit_code = await asyncio.wait_for(
                collect(), timeout=s.sandbox_timeout_s
            )
        except TimeoutError:
            timed_out = True
            await self._kill(name)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            out, out_trunc, err, err_trunc, exit_code = b"", False, b"", False, -1

        duration = time.perf_counter() - started
        stderr = err.decode("utf-8", errors="replace")
        if not timed_out and (
            exit_code == _DOCKER_DAEMON_ERROR
            or any(marker in stderr.lower() for marker in _DOCKER_FAILURE_MARKERS)
        ):
            hint = (
                f" Run: docker pull {s.sandbox_image}"
                if "no such image" in stderr.lower() or "unable to find image" in stderr.lower()
                else ""
            )
            raise SandboxError(f"docker run failed: {stderr.strip()[:300]}{hint}")

        max_chars = s.sandbox_max_output_chars
        stdout_text = out.decode("utf-8", errors="replace")
        truncated = (
            out_trunc or err_trunc or len(stdout_text) > max_chars or len(stderr) > max_chars
        )
        if exit_code == _OOM_KILLED:
            stderr += f"\n(killed: probably exceeded the {s.sandbox_memory_mb} MB memory limit)"
        log.info(
            "sandbox_run",
            exit_code=exit_code,
            timed_out=timed_out,
            duration_s=round(duration, 2),
            truncated=truncated,
        )
        return SandboxResult(
            stdout=stdout_text[:max_chars],
            # Keep the END of stderr: the traceback's last lines name the actual error.
            stderr=stderr[-max_chars:],
            exit_code=None if timed_out else exit_code,
            timed_out=timed_out,
            truncated=truncated,
            duration_s=duration,
        )
