import importlib.util
import os
import subprocess
import sys
from pathlib import Path


import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_setup_env_resolves_cross_platform_venv_python_paths():
    setup_env = _load_module("setup_env_under_test", ROOT / "setup_env.py")

    assert setup_env.venv_python_path(Path("/repo/.venv"), "win32") == Path(
        "/repo/.venv/Scripts/python.exe"
    )
    assert setup_env.venv_python_path(Path("/repo/.venv"), "darwin") == Path(
        "/repo/.venv/bin/python"
    )
    assert setup_env.venv_python_path(Path("/repo/.venv"), "linux") == Path(
        "/repo/.venv/bin/python"
    )


def test_setup_env_ffmpeg_is_required_by_default():
    setup_env = _load_module("setup_env_under_test", ROOT / "setup_env.py")

    assert setup_env.parse_args([]).require_ffmpeg is True
    assert setup_env.parse_args(["--dev"]).dev is True
    assert setup_env.user_python_command("darwin") == "python3"
    assert setup_env.user_python_command("linux") == "python3"
    assert setup_env.user_python_command("win32") == "py"
    assert "FFmpeg is required" in setup_env.ffmpeg_install_hint("darwin")
    assert "brew install ffmpeg" in setup_env.ffmpeg_install_hint("darwin")
    assert "winget install" in setup_env.ffmpeg_install_hint("win32")


def test_setup_env_check_probes_venv_python_version_and_required_imports(monkeypatch, tmp_path):
    setup_env = _load_module("setup_env_under_test", ROOT / "setup_env.py")
    venv_python = setup_env.venv_python_path(tmp_path / ".venv")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    calls = []
    monkeypatch.setattr(setup_env, "VENV_DIR", tmp_path / ".venv")
    monkeypatch.setattr(setup_env, "RUST_EDITOR_DIR", tmp_path / "missing-rust-editor")
    monkeypatch.setattr(setup_env, "ensure_ffmpeg", lambda: None)
    monkeypatch.setattr(setup_env, "ensure_python_version", lambda: None)
    monkeypatch.setattr(
        setup_env.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)) or subprocess.CompletedProcess(cmd, 0),
    )

    setup_env.check_environment(skip_rust=True)

    assert calls
    probe_cmd = calls[0][0]
    assert probe_cmd[:2] == [str(venv_python), "-c"]
    assert "PySide6" in probe_cmd[2]
    assert "cv2" in probe_cmd[2]
    assert "numpy" in probe_cmd[2]


def test_root_launcher_prefers_repo_venv_python(tmp_path):
    launcher = _load_module("run_launcher_under_test", ROOT / "run.py")
    venv_python = launcher.venv_python_path(tmp_path, "win32")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    assert launcher.find_venv_python(tmp_path, "win32") == venv_python
    assert launcher.should_reexec_into_venv(Path(sys.executable), venv_python)
    assert not launcher.should_reexec_into_venv(venv_python, venv_python)


def test_root_launcher_setup_message_points_to_single_setup_script():
    launcher = _load_module("run_launcher_under_test", ROOT / "run.py")

    message = launcher.setup_required_message()

    assert "python3 setup_env.py" in message
    assert "setup_windows.bat" not in message
    assert "setup.sh" not in message
    assert launcher.user_python_command("win32") == "py"


def test_root_launcher_main_fails_when_venv_is_missing(monkeypatch, capsys):
    launcher = _load_module("run_launcher_under_test", ROOT / "run.py")
    monkeypatch.setattr(launcher, "find_venv_python", lambda root_dir: None)

    with pytest.raises(SystemExit) as exc_info:
        launcher.main()

    assert exc_info.value.code == 1
    assert "python3 setup_env.py" in capsys.readouterr().err


def test_root_launcher_main_fails_when_ffmpeg_is_missing(monkeypatch, tmp_path, capsys):
    launcher = _load_module("run_launcher_under_test", ROOT / "run.py")
    venv_python = Path(sys.executable)
    monkeypatch.setattr(launcher, "find_venv_python", lambda root_dir: venv_python)
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher.runpy, "run_path", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not launch")))

    with pytest.raises(SystemExit) as exc_info:
        launcher.main()

    assert exc_info.value.code == 1
    assert "FFmpeg is required" in capsys.readouterr().err


def test_package_launcher_has_no_interactive_windows_pause():
    package_launcher = (ROOT / "synthesia2midi" / "run.py").read_text(encoding="utf-8")

    assert "input()" not in package_launcher


def test_package_launcher_imports_from_repo_root_pythonpath(tmp_path):
    script = """
import importlib.util
from pathlib import Path

path = Path("synthesia2midi/run.py").resolve()
spec = importlib.util.spec_from_file_location("package_launcher_under_test", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
"""

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["SYNTHESIA2MIDI_LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_find_ffmpeg_prefers_runtime_bundle(monkeypatch, tmp_path):
    from synthesia2midi.runtime_paths import RuntimePaths
    from synthesia2midi.utils import ffmpeg_helper

    ffmpeg = tmp_path / "bundle" / "bin" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_text("", encoding="utf-8")
    ffmpeg.chmod(0o755)

    monkeypatch.setattr(
        ffmpeg_helper,
        "detect_runtime_paths",
        lambda: RuntimePaths(
            frozen=True,
            app_root=tmp_path / "bundle",
            repo_root=tmp_path / "repo",
            home_dir=tmp_path / "home",
            platform_name="darwin",
        ),
    )
    monkeypatch.setattr(ffmpeg_helper.shutil, "which", lambda name: None)

    assert ffmpeg_helper.find_ffmpeg() == str(ffmpeg)


def test_find_ffprobe_prefers_runtime_bundle(monkeypatch, tmp_path):
    from synthesia2midi.runtime_paths import RuntimePaths
    from synthesia2midi.utils import ffmpeg_helper

    ffprobe = tmp_path / "bundle" / "bin" / "ffprobe"
    ffprobe.parent.mkdir(parents=True)
    ffprobe.write_text("", encoding="utf-8")
    ffprobe.chmod(0o755)

    monkeypatch.setattr(
        ffmpeg_helper,
        "detect_runtime_paths",
        lambda: RuntimePaths(
            frozen=True,
            app_root=tmp_path / "bundle",
            repo_root=tmp_path / "repo",
            home_dir=tmp_path / "home",
            platform_name="darwin",
        ),
    )
    monkeypatch.setattr(ffmpeg_helper.shutil, "which", lambda name: None)

    assert ffmpeg_helper.find_ffprobe() == str(ffprobe)
