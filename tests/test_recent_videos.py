from types import SimpleNamespace

from PySide6.QtWidgets import QApplication, QDialog, QPushButton

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


def test_startup_dialog_recent_rows_are_readable_and_keep_full_path_in_tooltip(tmp_path):
    QApplication.instance() or QApplication([])
    recent_path = tmp_path / "nested" / "song.mp4"
    recent_path.parent.mkdir()
    recent_path.write_text("video")
    dialog = StartupDialog(recent_video_paths=[str(recent_path)])

    try:
        recent_button = dialog.recent_video_buttons[0]

        assert type(recent_button) is QPushButton
        assert recent_button.text() == "song.mp4"
        assert str(recent_path.parent) not in recent_button.text()
        assert recent_button.minimumHeight() >= 36
        assert recent_button.toolTip() == str(recent_path)
    finally:
        dialog.deleteLater()
        QApplication.processEvents()


def test_startup_dialog_disambiguates_duplicate_filenames(tmp_path):
    QApplication.instance() or QApplication([])
    first = tmp_path / "one" / "song.mp4"
    second = tmp_path / "two" / "song.mp4"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("video")
    second.write_text("video")
    dialog = StartupDialog(recent_video_paths=[str(first), str(second)])
    try:
        assert dialog.recent_video_buttons[0].text() == "song.mp4 — one"
        assert dialog.recent_video_buttons[1].text() == "song.mp4 — two"
    finally:
        dialog.deleteLater()


def test_startup_dialog_middle_elides_long_recent_name_but_keeps_extension(tmp_path):
    QApplication.instance() or QApplication([])
    recent_path = tmp_path / (("a-very-long-piano-arrangement-title-" * 5) + ".mp4")
    recent_path.write_text("video")
    dialog = StartupDialog(recent_video_paths=[str(recent_path)])
    try:
        button = dialog.recent_video_buttons[0]
        assert "…" in button.text()
        assert button.text().endswith(".mp4")
        assert button.toolTip() == str(recent_path)
        assert button.accessibleDescription() == str(recent_path)
        assert button.fontMetrics().horizontalAdvance(button.text()) <= 430
    finally:
        dialog.deleteLater()


def test_startup_dialog_marks_missing_recent_video_instead_of_opening_it(tmp_path):
    QApplication.instance() or QApplication([])
    missing = tmp_path / "missing.mp4"
    dialog = StartupDialog(recent_video_paths=[str(missing)])
    try:
        button = dialog.recent_video_buttons[0]
        assert not button.isEnabled()
        assert button.text() == "missing.mp4 (missing)"
    finally:
        dialog.deleteLater()


def test_open_video_file_records_recent_only_after_success(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    selected_path = str(tmp_path / "picked.mp4")
    (tmp_path / "picked.mp4").write_text("video")

    created_dialogs = []

    class FakeFileDialog:
        ExistingFile = object()
        DontUseNativeDialog = object()

        def __init__(self, parent):
            self.parent = parent
            created_dialogs.append(self)

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
    parent_marker = object()

    result = VideoSessionUiController(app).open_video_file(parent=parent_marker)

    assert result is True
    assert created_dialogs[0].parent is parent_marker
    assert recent_store.paths == [selected_path]


def test_failed_file_picker_load_does_not_record_recent(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    selected_path = str(tmp_path / "failed.mp4")

    class FakeFileDialog:
        ExistingFile = object()
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

    result = VideoSessionUiController(app).open_video_file()

    assert result is False
    assert recent_store.paths == []


def test_open_video_file_returns_false_when_picker_is_cancelled(monkeypatch):
    from synthesia2midi.gui import video_session_ui_controller as module

    class RejectedFileDialog:
        ExistingFile = object()

        def __init__(self, parent):
            self.parent = parent

        def setWindowTitle(self, value):
            pass

        def setFileMode(self, value):
            pass

        def setNameFilter(self, value):
            pass

        def setDirectory(self, value):
            pass

        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(module, "QFileDialog", RejectedFileDialog)
    app = SimpleNamespace(recent_video_store=RecordingRecentStore())

    assert VideoSessionUiController(app).open_video_file() is False


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

    recent_loaded = controller.open_recent_video_file(recent_path)
    youtube_loaded = controller.handle_youtube_video_downloaded(youtube_path)

    assert recent_loaded is True
    assert youtube_loaded is True
    assert loaded_paths == [recent_path, youtube_path]
    assert recent_store.paths == [recent_path]


def test_recent_file_open_reports_failed_load(tmp_path):
    recent_path = str(tmp_path / "recent.mp4")
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: False
        ),
        recent_video_store=RecordingRecentStore(),
    )

    assert VideoSessionUiController(app).open_recent_video_file(recent_path) is False
    assert app.recent_video_store.paths == []


def test_image_sequence_uses_a_directory_picker(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    selected_dir = tmp_path / "frames"
    selected_dir.mkdir()
    picker_calls = []
    monkeypatch.setattr(
        module.QFileDialog,
        "getExistingDirectory",
        lambda parent, title, directory: picker_calls.append((title, directory))
        or str(selected_dir),
    )
    loaded = []
    recent_store = RecordingRecentStore()
    app = SimpleNamespace(
        video_session_coordinator=SimpleNamespace(
            load_path=lambda filepath, *, log_prefix, update_fps_display: loaded.append(
                (filepath, log_prefix, update_fps_display)
            )
            or True
        ),
        recent_video_store=recent_store,
    )

    VideoSessionUiController(app).open_image_sequence_folder()

    assert picker_calls
    assert picker_calls[0][0] == "Open Image Sequence Folder"
    assert loaded == [(str(selected_dir), "_open_image_sequence_folder", True)]
    assert recent_store.paths == [str(selected_dir)]
