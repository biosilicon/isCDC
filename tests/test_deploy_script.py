from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = PROJECT_ROOT / "deploy_test.sh"


def _run_script(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    process_env = os.environ.copy()
    process_env.update(env or {})
    return subprocess.run(
        ["bash", str(DEPLOY_SCRIPT), *arguments],
        cwd=PROJECT_ROOT,
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_tmux(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "tmux",
        "#!/bin/sh\n"
        'if [ "$1" = "has-session" ]; then\n'
        "    exit 1\n"
        "fi\n"
        "exit 99\n",
    )
    return bin_dir


def test_unknown_command_prints_usage() -> None:
    result = _run_script("unknown")

    assert result.returncode == 2
    assert "Usage: ./deploy_test.sh" in result.stderr
    assert "ISCDC_DEPLOY_START_TIMEOUT" in result.stderr


@pytest.mark.parametrize("value", ["0", "601", "invalid"])
def test_invalid_start_timeout_is_rejected(value: str) -> None:
    result = _run_script("status", env={"ISCDC_DEPLOY_START_TIMEOUT": value})

    assert result.returncode == 1
    assert "ISCDC_DEPLOY_START_TIMEOUT" in result.stderr


def test_start_reports_missing_python() -> None:
    result = _run_script("start", env={"ISCDC_PYTHON": "/missing/iscdc/python"})

    assert result.returncode == 1
    assert "iscdc Python executable is not available" in result.stderr


@pytest.mark.parametrize(
    ("command", "expected_code"),
    [("stop", 0), ("status", 1)],
)
def test_commands_report_when_service_is_not_running(
    tmp_path: Path, command: str, expected_code: int
) -> None:
    fake_bin = _fake_tmux(tmp_path)
    result = _run_script(
        command,
        env={
            "ISCDC_DEPLOY_SESSION": "iscdc_test_missing",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == expected_code
    assert "isCDC is not running" in result.stdout


def test_status_bypasses_proxy_for_local_health_check(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_arguments = tmp_path / "curl-arguments"
    _write_executable(bin_dir / "tmux", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "ss",
        "#!/bin/sh\nprintf 'LISTEN 0 128 0.0.0.0:5000 0.0.0.0:*\\n'\n",
    )
    _write_executable(
        bin_dir / "curl",
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_CURL_ARGUMENTS\"\n",
    )

    result = _run_script(
        "status",
        env={
            "FAKE_CURL_ARGUMENTS": str(curl_arguments),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0
    arguments = curl_arguments.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["--noproxy", "*"]
    assert arguments[-1] == "http://127.0.0.1:5000/healthz"


def test_stop_sends_interrupt_to_the_managed_session(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    running_marker = tmp_path / "running"
    running_marker.touch()
    tmux_arguments = tmp_path / "tmux-arguments"
    _write_executable(
        bin_dir / "tmux",
        "#!/bin/sh\n"
        'if [ "$1" = "has-session" ]; then\n'
        '    test -e "$FAKE_TMUX_RUNNING"\n'
        "    exit $?\n"
        "fi\n"
        'if [ "$1" = "send-keys" ]; then\n'
        '    printf \'%s\\n\' "$@" > "$FAKE_TMUX_ARGUMENTS"\n'
        '    rm -f "$FAKE_TMUX_RUNNING"\n'
        "    exit 0\n"
        "fi\n"
        "exit 99\n",
    )

    result = _run_script(
        "stop",
        env={
            "FAKE_TMUX_ARGUMENTS": str(tmux_arguments),
            "FAKE_TMUX_RUNNING": str(running_marker),
            "ISCDC_DEPLOY_SESSION": "iscdc_running_test",
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
    )

    assert result.returncode == 0
    assert tmux_arguments.read_text(encoding="utf-8").splitlines() == [
        "send-keys",
        "-t",
        "iscdc_running_test",
        "C-c",
    ]
