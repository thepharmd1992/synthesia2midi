from types import SimpleNamespace

from PySide6.QtWidgets import QMessageBox

from synthesia2midi.runtime_paths import RuntimePaths
from synthesia2midi.workflows.video_to_frames import VideoToFramesController


class FakeSignal:
    def __init__(self):
        self.connected = []

    def connect(self, callback):
        self.connected.append(callback)


class FakeButton:
    def __init__(self):
        self.enabled = []
        self.text = []

    def setEnabled(self, enabled):
        self.enabled.append(enabled)

    def setText(self, text):
        self.text.append(text)


class FakeWorker:
    instances = []

    def __init__(self, video_path, output_dir, quality=90):
        self.video_path = video_path
        self.output_dir = output_dir
        self.quality = quality
        self.progress_updated = FakeSignal()
        self.conversion_finished = FakeSignal()
        self.started = False
        FakeWorker.instances.append(self)

    def start(self):
        self.started = True


def _runtime_paths(tmp_path):
    return RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )


def _fake_app(video_path, original_video_path=None):
    return SimpleNamespace(
        app_state=SimpleNamespace(
            video=SimpleNamespace(
                filepath=str(video_path),
                original_video_path=str(original_video_path) if original_video_path else None,
            )
        ),
        control_panel=SimpleNamespace(video_to_frames_button=FakeButton()),
        video_to_frames_worker=None,
    )


def test_video_to_frames_controller_uses_project_frames_dir(tmp_path):
    runtime_paths = _runtime_paths(tmp_path)
    controller = VideoToFramesController(_fake_app(tmp_path / "song.mp4"), runtime_paths=runtime_paths)

    assert controller._frames_dir_for_video(str(tmp_path / "song.mp4")) == runtime_paths.project_frames_dir(
        str(tmp_path / "song.mp4")
    )


def test_video_to_frames_controller_reuses_existing_legacy_frames_dir(tmp_path):
    runtime_paths = _runtime_paths(tmp_path)
    video_path = tmp_path / "song.mp4"
    legacy_frames = tmp_path / "song_frames"
    legacy_frames.mkdir()
    controller = VideoToFramesController(_fake_app(video_path), runtime_paths=runtime_paths)

    assert controller._frames_dir_for_video(str(video_path)) == legacy_frames


def test_frame_series_loaded_from_project_data_uses_original_video_for_manual_conversion(monkeypatch, tmp_path):
    FakeWorker.instances.clear()
    runtime_paths = _runtime_paths(tmp_path)
    original_video = tmp_path / "song.mp4"
    original_video.write_bytes(b"video")
    project_frames = runtime_paths.project_frames_dir(str(original_video))
    project_frames.mkdir(parents=True)
    app = _fake_app(project_frames, original_video_path=original_video)
    info_calls = []

    monkeypatch.setattr(
        "synthesia2midi.workflows.video_to_frames.check_ffmpeg_available",
        lambda: (True, "ok"),
        raising=False,
    )
    monkeypatch.setattr(
        "synthesia2midi.utils.ffmpeg_helper.check_ffmpeg_available",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        "synthesia2midi.workflows.video_to_frames.QMessageBox.information",
        lambda *args: info_calls.append(args),
    )
    monkeypatch.setattr(
        "synthesia2midi.workflows.video_to_frames.QMessageBox.question",
        lambda *args: QMessageBox.Yes,
    )
    monkeypatch.setattr("synthesia2midi.workflows.video_to_frames.VideoToFramesWorker", FakeWorker)

    VideoToFramesController(app, runtime_paths=runtime_paths).handle_request()

    worker = FakeWorker.instances[0]
    assert worker.video_path == str(original_video)
    assert worker.output_dir == str(project_frames)
    assert worker.started is True
    assert info_calls
