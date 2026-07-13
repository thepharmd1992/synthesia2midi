import ast
import importlib.util
from pathlib import Path

from PySide6.QtWidgets import QDialog

from synthesia2midi.runtime_paths import RuntimePaths


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _main_guard_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(part, ast.Constant) and part.value == "__main__"
            for part in ast.walk(node.test)
        )
    )
    return {
        ast.unparse(node.func)
        for node in ast.walk(guard)
        if isinstance(node, ast.Call)
    }


def test_gui_launch_guards_begin_startup_without_showing_main():
    paths = [
        ROOT / "synthesia2midi" / "run.py",
        ROOT / "synthesia2midi" / "synthesia2midi" / "main.py",
    ]

    for path in paths:
        calls = _main_guard_calls(path)
        assert "app.begin_startup" in calls
        assert "app.show" not in calls


def test_default_log_dir_uses_runtime_path_when_frozen(monkeypatch, tmp_path):
    from synthesia2midi.core import logging_config

    fake_paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )
    monkeypatch.setattr(logging_config, "detect_runtime_paths", lambda: fake_paths)

    assert logging_config._default_log_dir() == str(fake_paths.log_dir())


def test_package_launcher_missing_dependency_message_changes_when_frozen(tmp_path):
    launcher = _load_module("package_launcher_under_test", ROOT / "synthesia2midi" / "run.py")
    launcher.runtime_paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )

    message = launcher._missing_dep_instructions("PySide6")

    assert "corrupted" in message
    assert "setup_env.py" not in message


def test_youtube_dialog_uses_runtime_default_download_dir(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    calls = {}

    class _Signal:
        def connect(self, callback):
            calls["connected"] = callback

    class FakeDialog:
        def __init__(self, parent=None, default_output_dir=""):
            calls["parent"] = parent
            calls["default_output_dir"] = default_output_dir
            self.video_downloaded = _Signal()

        def exec(self):
            return QDialog.Rejected

    fake_paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )
    (fake_paths.home_dir / "Documents").mkdir(parents=True)

    monkeypatch.setattr(module, "YouTubeDownloadDialog", FakeDialog)
    monkeypatch.setattr(module, "detect_runtime_paths", lambda: fake_paths)

    controller = module.VideoSessionUiController(app=object())
    parent_marker = object()
    result = controller.show_youtube_download_dialog(parent=parent_marker)

    assert result is False
    assert calls["parent"] is parent_marker
    assert calls["default_output_dir"] == str(fake_paths.default_download_dir())


def test_youtube_dialog_returns_true_after_downloaded_video_loads(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from synthesia2midi.gui import video_session_ui_controller as module

    calls = {}

    class _Signal:
        def connect(self, callback):
            calls["connected"] = callback

    class FakeDialog:
        def __init__(self, parent=None, default_output_dir=""):
            self.video_downloaded = _Signal()

        def exec(self):
            calls["connected"]("/tmp/downloaded.mp4")
            return QDialog.Accepted

    fake_paths = RuntimePaths(
        frozen=True,
        app_root=tmp_path / "bundle",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )
    loaded = []
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: loaded.append(
                filepath
            )
            or True
        )
    )
    monkeypatch.setattr(module, "YouTubeDownloadDialog", FakeDialog)
    monkeypatch.setattr(module, "detect_runtime_paths", lambda: fake_paths)

    result = module.VideoSessionUiController(app).show_youtube_download_dialog()

    assert result is True
    assert loaded == ["/tmp/downloaded.mp4"]
