from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QDialog

from synthesia2midi.core.recent_videos import RecentVideoStore
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.video_session_ui_controller import VideoSessionUiController


class MemorySettings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value


class RecordingRecentStore:
    def __init__(self):
        self.paths = []

    def add(self, path):
        self.paths.append(path)


def test_recent_video_store_promotes_existing_paths_and_keeps_five(tmp_path):
    settings = MemorySettings()
    store = RecentVideoStore(settings=settings)
    paths = []
    for index in range(6):
        path = tmp_path / f"video-{index}.mp4"
        path.write_text("video")
        paths.append(str(path))

    for path in paths:
        store.add(path)
    store.add(paths[1])

    assert store.recent_paths() == [
        paths[1],
        paths[5],
        paths[4],
        paths[3],
        paths[2],
    ]


def test_recent_video_store_hides_missing_paths(tmp_path):
    existing = tmp_path / "existing.mp4"
    missing = tmp_path / "missing.mp4"
    existing.write_text("video")
    settings = MemorySettings({"recent_video_files": [str(missing), str(existing)]})
    store = RecentVideoStore(settings=settings)

    assert store.recent_paths() == [str(existing)]
    assert settings.values["recent_video_files"] == [str(existing)]


def test_startup_dialog_emits_recent_file_before_closing(tmp_path):
    QApplication.instance() or QApplication([])
    recent_path = tmp_path / "song.mp4"
    recent_path.write_text("video")
    dialog = StartupDialog(recent_video_paths=[str(recent_path)])
    emitted = []

    try:
        dialog.open_recent_file.connect(lambda path: emitted.append((path, dialog.result())))

        dialog.recent_video_buttons[0].click()

        assert emitted == [(str(recent_path), QDialog.Accepted)]
        assert dialog.result() == QDialog.Accepted
    finally:
        dialog.deleteLater()
        QApplication.processEvents()


def test_open_video_file_records_recent_only_after_success(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    selected_path = str(tmp_path / "picked.mp4")
    (tmp_path / "picked.mp4").write_text("video")

    class FakeFileDialog:
        AnyFile = object()
        DontUseNativeDialog = object()

        def __init__(self, parent):
            self.parent = parent

        def setWindowTitle(self, value):
            pass

        def setFileMode(self, value):
            pass

        def setOption(self, option, value):
            pass

        def setNameFilter(self, value):
            pass

        def setDirectory(self, value):
            pass

        def findChild(self, view_type):
            return None

        def exec(self):
            return QDialog.Accepted

        def selectedFiles(self):
            return [selected_path]

    recent_store = RecordingRecentStore()
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: True
        ),
        recent_video_store=recent_store,
    )
    monkeypatch.setattr(module, "QFileDialog", FakeFileDialog)

    VideoSessionUiController(app).open_video_file()

    assert recent_store.paths == [selected_path]


def test_failed_file_picker_load_does_not_record_recent(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    selected_path = str(tmp_path / "failed.mp4")

    class FakeFileDialog:
        AnyFile = object()
        DontUseNativeDialog = object()

        def __init__(self, parent):
            self.parent = parent

        def setWindowTitle(self, value):
            pass

        def setFileMode(self, value):
            pass

        def setOption(self, option, value):
            pass

        def setNameFilter(self, value):
            pass

        def setDirectory(self, value):
            pass

        def findChild(self, view_type):
            return None

        def exec(self):
            return QDialog.Accepted

        def selectedFiles(self):
            return [selected_path]

    recent_store = RecordingRecentStore()
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: False
        ),
        recent_video_store=recent_store,
    )
    monkeypatch.setattr(module, "QFileDialog", FakeFileDialog)

    VideoSessionUiController(app).open_video_file()

    assert recent_store.paths == []


def test_recent_click_records_promotion_but_youtube_download_does_not(tmp_path):
    recent_path = str(tmp_path / "recent.mp4")
    youtube_path = str(tmp_path / "youtube.mp4")
    recent_store = RecordingRecentStore()
    loaded_paths = []
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: loaded_paths.append(filepath) or True
        ),
        recent_video_store=recent_store,
    )
    controller = VideoSessionUiController(app)

    controller.open_recent_video_file(recent_path)
    controller.handle_youtube_video_downloaded(youtube_path)

    assert loaded_paths == [recent_path, youtube_path]
    assert recent_store.paths == [recent_path]
