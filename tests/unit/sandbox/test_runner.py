from __future__ import annotations

import pytest

from agentic_rag.config import Settings
from agentic_rag.core.exceptions import SandboxError
from agentic_rag.sandbox import DockerSandbox
from fakes import FakeProcess, FakeSpawner


def sandbox(settings: Settings, spawner: FakeSpawner, **update: object) -> DockerSandbox:
    s = settings.model_copy(update=update) if update else settings
    return DockerSandbox(s, process_factory=spawner)


def test_command_applies_every_isolation_flag(settings: Settings) -> None:
    cmd = DockerSandbox(settings).build_command("box-1")
    joined = " ".join(cmd)

    assert cmd[:3] == ["docker", "run", "--rm"]
    for flag in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--memory 256m",
        "--memory-swap 256m",
        "--pids-limit 64",
        "--name box-1",
    ):
        assert flag in joined, flag
    assert "-v" not in cmd
    assert "--volume" not in cmd
    assert cmd[-3:] == ["python", "-I", "-"]
    assert cmd.index(settings.sandbox_image) < cmd.index("python")


async def test_successful_run_returns_output_and_sends_code_on_stdin(settings: Settings) -> None:
    spawner = FakeSpawner(FakeProcess(stdout=b"42\n"))
    result = await sandbox(settings, spawner).run("print(6*7)")

    assert result.ok
    assert result.stdout == "42\n"
    assert result.exit_code == 0
    assert spawner.spawned[0].stdin.data == b"print(6*7)"
    assert spawner.spawned[0].stdin.closed


async def test_failing_script_is_a_result_not_an_error(settings: Settings) -> None:
    err = b"Traceback (most recent call last):\n  File ...\nZeroDivisionError: division by zero\n"
    result = await sandbox(settings, FakeSpawner(FakeProcess(stderr=err, returncode=1))).run("1/0")

    assert not result.ok
    assert result.exit_code == 1
    assert "ZeroDivisionError" in result.stderr


async def test_timeout_kills_the_container(settings: Settings) -> None:
    hanging = FakeProcess(hang=True)
    spawner = FakeSpawner(hanging)
    result = await sandbox(settings, spawner, sandbox_timeout_s=0.05).run("while True: pass")

    assert result.timed_out
    assert not result.ok
    assert result.exit_code is None
    assert hanging.killed
    kill_calls = [c for c in spawner.calls if c[:2] == ("docker", "kill")]
    assert len(kill_calls) == 1
    run_call = spawner.calls[0]
    assert kill_calls[0][2] == run_call[run_call.index("--name") + 1]  # same container name


async def test_output_flood_is_truncated_not_buffered(settings: Settings) -> None:
    flood = b"x" * 1_000_000
    result = await sandbox(
        settings, FakeSpawner(FakeProcess(stdout=flood)), sandbox_max_output_chars=500
    ).run("print('x' * 10**6)")

    assert result.truncated
    assert len(result.stdout) == 500
    assert result.ok


async def test_stderr_keeps_the_tail(settings: Settings) -> None:
    err = b"a" * 5000 + b"\nValueError: the real problem"
    result = await sandbox(
        settings, FakeSpawner(FakeProcess(stderr=err, returncode=1)), sandbox_max_output_chars=300
    ).run("x")
    assert result.stderr.endswith("ValueError: the real problem")


async def test_oom_kill_is_explained(settings: Settings) -> None:
    result = await sandbox(settings, FakeSpawner(FakeProcess(returncode=137))).run("x")
    assert "memory limit" in result.stderr


async def test_docker_daemon_failure_raises(settings: Settings) -> None:
    proc = FakeProcess(stderr=b"docker: Cannot connect to the Docker daemon", returncode=125)
    with pytest.raises(SandboxError, match="Cannot connect"):
        await sandbox(settings, FakeSpawner(proc)).run("print(1)")


async def test_missing_docker_binary_raises_actionable_error(settings: Settings) -> None:
    box = sandbox(settings, FakeSpawner(missing=True))
    with pytest.raises(SandboxError, match="docker"):
        await box.run("print(1)")
    with pytest.raises(SandboxError, match="Docker Desktop"):
        await box.check()


async def test_check_passes_when_image_present(settings: Settings) -> None:
    await sandbox(settings, FakeSpawner(FakeProcess(stdout=b"[{}]"))).check()


async def test_check_explains_missing_image(settings: Settings) -> None:
    proc = FakeProcess(stdout=b"Error: No such image: python:3.12-slim", returncode=1)
    with pytest.raises(SandboxError, match=r"docker pull python:3\.12-slim"):
        await sandbox(settings, FakeSpawner(proc)).check()


async def test_check_explains_daemon_down(settings: Settings) -> None:
    proc = FakeProcess(stdout=b"error during connect: dial pipe", returncode=1)
    with pytest.raises(SandboxError, match="Docker Desktop running"):
        await sandbox(settings, FakeSpawner(proc)).check()


async def test_new_cli_daemon_down_message_is_detected_even_with_exit_code_1(
    settings: Settings,
) -> None:
    msg = b"failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine"
    with pytest.raises(SandboxError, match="docker run failed"):
        await sandbox(settings, FakeSpawner(FakeProcess(stderr=msg, returncode=1))).run("print(1)")


async def test_missing_image_on_run_suggests_pull(settings: Settings) -> None:
    msg = b"Unable to find image 'python:3.12-slim' locally"
    with pytest.raises(SandboxError, match=r"docker pull python:3\.12-slim"):
        await sandbox(settings, FakeSpawner(FakeProcess(stderr=msg, returncode=125))).run("x")


def test_command_never_pulls_images_implicitly(settings: Settings) -> None:
    cmd = DockerSandbox(settings).build_command("n")
    assert cmd[cmd.index("--pull") + 1] == "never"
