import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.assisted_calibration import (
    overlay_key_color,
    overlay_note_label,
    sample_overlay_bgr,
    sample_overlay_rgb,
)


def _overlay(
    key_id=1,
    note="C",
    octave=4,
    x=1,
    y=1,
    width=3,
    height=2,
    key_type="LW",
):
    return OverlayConfig(
        key_id=key_id,
        note_octave=octave,
        note_name_in_octave=note,
        x=x,
        y=y,
        width=width,
        height=height,
        key_type=key_type,
    )


def test_overlay_sampling_uses_clipped_integer_roi():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)
    frame[1:3, 1:4] = (10, 20, 30)

    assert sample_overlay_rgb(frame, _overlay()) == (10, 20, 30)
    assert sample_overlay_bgr(frame, _overlay()).mean(axis=(0, 1)).astype(int).tolist() == [30, 20, 10]


def test_overlay_sampling_returns_none_for_empty_roi():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)

    assert sample_overlay_rgb(frame, _overlay(x=99, y=99)) is None
    assert sample_overlay_bgr(frame, _overlay(x=99, y=99)) is None


def test_overlay_note_label_and_key_color_use_existing_overlay_data():
    assert overlay_note_label(_overlay(note="E", octave=4)) == "E4"
    assert overlay_key_color(_overlay(key_type="LB")) == "B"
    assert overlay_key_color(_overlay(key_type="RW")) == "W"
