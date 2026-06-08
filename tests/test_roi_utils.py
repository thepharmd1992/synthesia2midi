import numpy as np
import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.roi_utils import (
    adjust_overlay_for_crop,
    extract_roi_bgr,
    rotated_overlay_corners,
)


def make_overlay(**overrides) -> OverlayConfig:
    data = dict(
        key_id=7,
        note_octave=4,
        note_name_in_octave="C",
        x=10,
        y=20,
        width=30,
        height=40,
        unlit_reference_color=(1, 2, 3),
        key_type="white",
        overlay_type="key",
    )
    data.update(overrides)
    return OverlayConfig(**data)


def test_extract_roi_bgr_clips_partial_overlay_to_frame_bounds():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = make_overlay(x=8, y=7, width=10, height=10)

    roi = extract_roi_bgr(frame, overlay)

    assert roi is not None
    assert roi.shape == (3, 2, 3)


def test_extract_roi_bgr_returns_none_when_overlay_outside_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    overlay = make_overlay(x=20, y=20, width=5, height=5)

    assert extract_roi_bgr(frame, overlay) is None


def test_rotated_overlay_corners_pivot_around_overlay_center():
    overlay = make_overlay(x=10, y=20, width=4, height=2, rotation_degrees=90)

    corners = rotated_overlay_corners(overlay)

    assert corners == pytest.approx([
        (13, 19),
        (13, 23),
        (11, 23),
        (11, 19),
    ])


def test_extract_roi_bgr_samples_only_rotated_overlay_pixels():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    overlay = make_overlay(x=5, y=5, width=10, height=4, rotation_degrees=45)
    import cv2

    polygon = np.array(rotated_overlay_corners(overlay), dtype=np.int32)
    cv2.fillConvexPoly(frame, polygon, (20, 40, 60))
    frame[5:9, 5:15] = (200, 200, 200)
    cv2.fillConvexPoly(frame, polygon, (20, 40, 60))

    roi = extract_roi_bgr(frame, overlay)

    assert roi is not None
    assert roi.shape[2] == 3
    assert tuple(np.round(np.mean(roi, axis=(0, 1))).astype(int)) == (20, 40, 60)


def test_adjust_overlay_for_crop_preserves_note_fields_and_runtime_state():
    overlay = make_overlay(
        x=10,
        y=20,
        width=30,
        height=40,
        prev_progression_ratio=0.1,
        last_progression_ratio=0.2,
        last_is_lit=True,
        in_forced_delta_off_state=True,
    )

    adjusted = adjust_overlay_for_crop(overlay, crop_offset_x=3, crop_offset_y=5)

    assert adjusted is not None
    assert adjusted.key_id == overlay.key_id
    assert adjusted.note_octave == overlay.note_octave
    assert adjusted.note_name_in_octave == overlay.note_name_in_octave
    assert adjusted.x == 7
    assert adjusted.y == 15
    assert adjusted.width == overlay.width
    assert adjusted.height == overlay.height
    assert adjusted.unlit_reference_color == overlay.unlit_reference_color
    assert adjusted.key_type == overlay.key_type
    assert adjusted.overlay_type == overlay.overlay_type
    assert adjusted.rotation_degrees == overlay.rotation_degrees
    assert adjusted.prev_progression_ratio == overlay.prev_progression_ratio
    assert adjusted.last_progression_ratio == overlay.last_progression_ratio
    assert adjusted.last_is_lit == overlay.last_is_lit
    assert adjusted.in_forced_delta_off_state == overlay.in_forced_delta_off_state


def test_adjust_overlay_for_crop_handles_none_note_fields_and_preserves_histograms():
    unlit_hist = np.array([0.25, 0.75], dtype=np.float32)
    lit_hist = np.array([0.1, 0.9], dtype=np.float32)
    overlay = make_overlay(
        note_octave=None,
        note_name_in_octave=None,
        x=15,
        y=25,
        unlit_hist=unlit_hist,
        lit_hist=lit_hist,
        overlay_type="spark",
        in_forced_delta_off_state=True,
    )

    adjusted = adjust_overlay_for_crop(overlay, crop_offset_x=5, crop_offset_y=10)

    assert adjusted is not None
    assert adjusted.note_octave is None
    assert adjusted.note_name_in_octave is None
    assert adjusted.x == 10
    assert adjusted.y == 15
    assert adjusted.unlit_hist is unlit_hist
    assert adjusted.lit_hist is lit_hist
    assert adjusted.overlay_type == "spark"
    assert adjusted.in_forced_delta_off_state is True


def test_adjust_overlay_for_crop_returns_none_when_overlay_is_before_crop():
    overlay = make_overlay(x=2, y=3, width=4, height=5)

    assert adjust_overlay_for_crop(overlay, crop_offset_x=10, crop_offset_y=12) is None
