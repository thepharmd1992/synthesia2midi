import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.runtime_paths import RuntimePaths


def _runtime_paths(tmp_path):
    return RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path / "home",
        platform_name="darwin",
    )


def test_config_manager_round_trips_overlay_rotation(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    app_state = AppState()
    app_state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=10,
            y=20,
            width=30,
            height=40,
            key_type="LW",
            rotation_degrees=12.5,
        )
    ]

    runtime_paths = _runtime_paths(tmp_path)
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state, runtime_paths=runtime_paths)
    assert loaded_manager.load_config(str(runtime_paths.project_ini_path(str(video_path)))) is True

    assert len(loaded_state.overlays) == 1
    assert loaded_state.overlays[0].rotation_degrees == 12.5


def test_config_manager_round_trips_overlay_generation_source(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    app_state = AppState()
    app_state.calibration.overlay_generation_source = "manual"
    app_state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=10,
            y=20,
            width=30,
            height=40,
            key_type="LW",
        )
    ]

    runtime_paths = _runtime_paths(tmp_path)
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state, runtime_paths=runtime_paths)
    assert loaded_manager.load_config(str(runtime_paths.project_ini_path(str(video_path)))) is True

    assert loaded_state.calibration.overlay_generation_source == "manual"


def test_config_manager_round_trips_manual_keyboard_box(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")
    app_state = AppState()
    app_state.calibration.manual_keyboard_box = (10.0, 20.0, 300.0, 400.0)
    app_state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=10,
            y=20,
            width=30,
            height=40,
            key_type="LW",
        )
    ]

    runtime_paths = _runtime_paths(tmp_path)
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state, runtime_paths=runtime_paths)
    assert loaded_manager.load_config(str(runtime_paths.project_ini_path(str(video_path)))) is True

    assert loaded_state.calibration.manual_keyboard_box == pytest.approx((10.0, 20.0, 300.0, 400.0))


def test_save_config_writes_project_ini_and_overlay(tmp_path):
    video_path = tmp_path / "song.mp4"
    video_path.write_bytes(b"")
    runtime_paths = _runtime_paths(tmp_path)
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)

    assert manager.save_config(str(video_path)) is True

    assert runtime_paths.project_ini_path(str(video_path)).exists()
    assert runtime_paths.project_overlay_json_path(str(video_path)).exists()
    assert not (tmp_path / "song.ini").exists()
    assert not (tmp_path / "song_overlays.json").exists()


def test_load_config_falls_back_to_legacy_sidecar(tmp_path):
    video_path = tmp_path / "song.mp4"
    video_path.write_bytes(b"")
    legacy_ini = tmp_path / "song.ini"
    legacy_overlay = tmp_path / "song_overlays.json"
    legacy_ini.write_text(
        "[Video]\n"
        f"filepath = {video_path}\n\n"
        "[Settings]\n"
        "overlay_generation_source = manual\n"
        "manual_keyboard_box = 10,20,300,400\n",
        encoding="utf-8",
    )
    legacy_overlay.write_text(
        '{"overlays": [], "exemplar_lit_histograms": {}, "overlay_generation_source": "manual", "manual_keyboard_box": [10, 20, 300, 400]}',
        encoding="utf-8",
    )
    runtime_paths = _runtime_paths(tmp_path)
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=runtime_paths)

    assert legacy_ini in manager.config_candidates_for_video(str(video_path))
    assert manager.load_config(str(legacy_ini)) is True
    assert app_state.calibration.manual_keyboard_box == pytest.approx((10.0, 20.0, 300.0, 400.0))
