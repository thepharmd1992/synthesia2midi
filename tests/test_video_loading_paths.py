import pytest
from types import SimpleNamespace

from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.runtime_paths import RuntimePaths
from synthesia2midi.workflows.video_loading import VideoLoadingWorkflow


def _runtime_paths(tmp_path):
    return RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )


def _workflow(tmp_path):
    runtime_paths = _runtime_paths(tmp_path)
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)
    workflow = VideoLoadingWorkflow(app_state, manager, runtime_paths=runtime_paths)
    return runtime_paths, app_state, manager, workflow


def _write_minimal_ini(path, video_path, manual_keyboard_box):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Video]\n"
        f"filepath = {video_path}\n\n"
        "[Settings]\n"
        "overlay_generation_source = manual\n"
        f"manual_keyboard_box = {manual_keyboard_box}\n",
        encoding="utf-8",
    )


def test_load_associated_config_prefers_project_config(tmp_path):
    runtime_paths, app_state, _manager, workflow = _workflow(tmp_path)
    video_path = tmp_path / "song.mp4"
    video_path.write_bytes(b"")
    project_ini = runtime_paths.project_ini_path(str(video_path))
    legacy_ini = tmp_path / "song.ini"
    _write_minimal_ini(project_ini, video_path, "1,2,30,40")
    _write_minimal_ini(legacy_ini, video_path, "10,20,300,400")

    assert workflow._load_associated_config(str(video_path)) is True

    assert app_state.video.filepath_ini_used == str(project_ini)
    assert app_state.calibration.manual_keyboard_box == pytest.approx((1.0, 2.0, 30.0, 40.0))


def test_load_associated_config_falls_back_to_legacy_config(tmp_path):
    _runtime_paths, app_state, _manager, workflow = _workflow(tmp_path)
    video_path = tmp_path / "song.mp4"
    video_path.write_bytes(b"")
    legacy_ini = tmp_path / "song.ini"
    _write_minimal_ini(legacy_ini, video_path, "10,20,300,400")

    assert workflow._load_associated_config(str(video_path)) is True

    assert app_state.video.filepath_ini_used == str(legacy_ini)
    assert app_state.calibration.manual_keyboard_box == pytest.approx((10.0, 20.0, 300.0, 400.0))


def test_save_current_config_records_project_ini_path(tmp_path):
    runtime_paths, app_state, _manager, workflow = _workflow(tmp_path)
    video_path = tmp_path / "song.mp4"
    video_path.write_bytes(b"")
    app_state.video.filepath = str(video_path)

    assert workflow.save_current_config() is True

    project_ini = runtime_paths.project_ini_path(str(video_path))
    assert project_ini.exists()
    assert app_state.video.filepath_ini_used == str(project_ini)


def test_frame_conversion_uses_project_frames_dir(tmp_path):
    runtime_paths, _app_state, _manager, workflow = _workflow(tmp_path)
    video_path = tmp_path / "song.mp4"

    assert workflow._frames_dir_for_video(str(video_path)) == runtime_paths.project_frames_dir(str(video_path))


def test_existing_legacy_frames_dir_is_reused(tmp_path):
    _runtime_paths, _app_state, _manager, workflow = _workflow(tmp_path)
    video_path = tmp_path / "song.mp4"
    legacy_frames = tmp_path / "song_frames"
    legacy_frames.mkdir()

    assert workflow._frames_dir_for_video(str(video_path)) == legacy_frames


def test_loading_or_closing_video_invalidates_alignment_review(tmp_path):
    _runtime_paths, app_state, _manager, workflow = _workflow(tmp_path)
    app_state.calibration.alignment_reviewed = True
    session = SimpleNamespace(fps=30.0, total_frames=120)

    workflow._update_video_state(str(tmp_path / "next.mp4"), session)

    assert app_state.calibration.alignment_reviewed is False

    app_state.calibration.alignment_reviewed = True
    workflow.close_video()

    assert app_state.calibration.alignment_reviewed is False
