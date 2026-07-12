import configparser

import numpy as np
import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState
from synthesia2midi.core.color_families import SUPPORTED_EXEMPLAR_SLOTS
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


def test_old_ini_without_dynamic_enabled_section_preserves_legacy_defaults(tmp_path):
    old_ini = tmp_path / "old.ini"
    old_ini.write_text(
        "[Settings]\n"
        "exemplar_enabled_lw = false\n"
        "exemplar_enabled_lb = true\n"
        "exemplar_enabled_rw = false\n"
        "exemplar_enabled_rb = true\n",
        encoding="utf-8",
    )
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=_runtime_paths(tmp_path))

    assert manager.load_config(str(old_ini)) is True
    assert app_state.detection.exemplar_key_type_enabled == {
        "LW": False,
        "LB": True,
        "RW": False,
        "RB": True,
        "COLOR_3_W": False,
        "COLOR_3_B": False,
        "COLOR_4_W": False,
        "COLOR_4_B": False,
    }


def test_four_family_enabled_flags_colors_and_histograms_round_trip(tmp_path):
    video_path = tmp_path / "four-families.mp4"
    video_path.write_bytes(b"")
    runtime_paths = _runtime_paths(tmp_path)
    app_state = AppState()
    app_state.detection.exemplar_key_type_enabled["COLOR_3_W"] = True
    app_state.detection.exemplar_key_type_enabled["COLOR_4_B"] = True
    app_state.detection.exemplar_lit_colors["COLOR_3_W"] = (12, 34, 56)
    app_state.detection.exemplar_lit_colors["COLOR_4_B"] = (78, 90, 123)
    app_state.detection.exemplar_lit_histograms["COLOR_3_W"] = np.array(
        [0.0, 0.25, 1.0], dtype=np.float32
    )
    app_state.detection.exemplar_lit_histograms["COLOR_4_B"] = np.array(
        [2.0, 3.5, 4.0], dtype=np.float64
    )

    manager = ConfigManager(app_state, runtime_paths=runtime_paths)
    assert manager.save_config(str(video_path)) is True

    parser = configparser.ConfigParser()
    parser.read(runtime_paths.project_ini_path(str(video_path)), encoding="utf-8")
    assert {
        slot.upper(): parser.getboolean("ExemplarEnabled", slot)
        for slot in SUPPORTED_EXEMPLAR_SLOTS
    } == app_state.detection.exemplar_key_type_enabled

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state, runtime_paths=runtime_paths)
    assert loaded_manager.load_config(
        str(runtime_paths.project_ini_path(str(video_path)))
    ) is True
    assert loaded_state.detection.exemplar_key_type_enabled["COLOR_3_W"] is True
    assert loaded_state.detection.exemplar_key_type_enabled["COLOR_4_B"] is True
    assert loaded_state.detection.exemplar_lit_colors["COLOR_3_W"] == (12, 34, 56)
    assert loaded_state.detection.exemplar_lit_colors["COLOR_4_B"] == (78, 90, 123)
    np.testing.assert_array_equal(
        loaded_state.detection.exemplar_lit_histograms["COLOR_3_W"],
        np.array([0.0, 0.25, 1.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        loaded_state.detection.exemplar_lit_histograms["COLOR_4_B"],
        np.array([2.0, 3.5, 4.0], dtype=np.float64),
    )


def test_four_family_enabled_ignores_invalid_color_three_sample(tmp_path):
    config_path = tmp_path / "invalid-color-three.ini"
    config_path.write_text(
        "[Settings]\n\n"
        "[ExemplarEnabled]\n"
        "color_3_w = true\n"
        "color_3_b = true\n\n"
        "[ExemplarLitColors]\n"
        "color_3_w = not,a,color\n"
        "color_3_b = 4,5,6\n",
        encoding="utf-8",
    )
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=_runtime_paths(tmp_path))

    assert manager.load_config(str(config_path)) is True
    assert app_state.detection.exemplar_lit_colors["COLOR_3_W"] is None
    assert app_state.detection.exemplar_lit_colors["COLOR_3_B"] == (4, 5, 6)
    assert app_state.detection.get_required_exemplar_types() == [
        "LW",
        "LB",
        "RW",
        "RB",
        "COLOR_3_W",
        "COLOR_3_B",
    ]
