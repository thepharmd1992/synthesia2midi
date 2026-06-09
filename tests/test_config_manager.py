import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.core.app_state import AppState


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

    manager = ConfigManager(app_state)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state)
    assert loaded_manager.load_config(str(tmp_path / "sample.ini")) is True

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

    manager = ConfigManager(app_state)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state)
    assert loaded_manager.load_config(str(tmp_path / "sample.ini")) is True

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

    manager = ConfigManager(app_state)

    assert manager.save_config(str(video_path)) is True

    loaded_state = AppState()
    loaded_manager = ConfigManager(loaded_state)
    assert loaded_manager.load_config(str(tmp_path / "sample.ini")) is True

    assert loaded_state.calibration.manual_keyboard_box == pytest.approx((10.0, 20.0, 300.0, 400.0))
