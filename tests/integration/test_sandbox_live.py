"""Real Docker sandbox. Run with: pytest -m integration  (needs: docker pull python:3.12-slim)"""

from __future__ import annotations

import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import SandboxError
from agentic_rag.sandbox import DockerSandbox

pytestmark = pytest.mark.integration


@pytest.fixture
async def box() -> DockerSandbox:
    sandbox = DockerSandbox(Settings(_env_file=None, sandbox_timeout_s=8))
    try:
        await sandbox.check()
    except SandboxError as exc:
        pytest.skip(str(exc))
    return sandbox


async def test_runs_code(box: DockerSandbox) -> None:
    result = await box.run("print(6 * 7)")
    assert result.ok
    assert result.stdout.strip() == "42"


async def test_has_no_network(box: DockerSandbox) -> None:
    code = "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)\n"
    result = await box.run(code)
    assert not result.ok
    assert "OSError" in result.stderr or "Error" in result.stderr


async def test_root_filesystem_is_read_only(box: DockerSandbox) -> None:
    result = await box.run("open('/etc/pwned', 'w')")
    assert not result.ok


async def test_infinite_loop_is_killed(box: DockerSandbox) -> None:
    result = await box.run("while True: pass")
    assert result.timed_out
