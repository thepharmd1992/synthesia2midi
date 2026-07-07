import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.assisted_calibration import (
    ExemplarScanSettings,
    assess_unlit_frame,
    capture_unlit_references_from_frame,
    overlay_key_color,
    overlay_note_label,
    sample_overlay_bgr,
    sample_overlay_rgb,
    scan_lit_exemplar_candidates,
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


def test_overlay_sampling_truncates_fractional_overlay_bounds():
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    overlay = _overlay(x=1.6, y=1.6, width=1.6, height=1.6)

    assert sample_overlay_rgb(frame, overlay) == tuple(frame[1, 1])
    assert sample_overlay_bgr(frame, overlay).shape == (1, 1, 3)
    assert sample_overlay_bgr(frame, overlay)[0, 0].tolist() == frame[1, 1][::-1].tolist()


def test_overlay_sampling_returns_none_for_empty_roi():
    frame = np.zeros((5, 6, 3), dtype=np.uint8)

    assert sample_overlay_rgb(frame, _overlay(x=99, y=99)) is None
    assert sample_overlay_bgr(frame, _overlay(x=99, y=99)) is None


def test_overlay_note_label_and_key_color_use_existing_overlay_data():
    assert overlay_note_label(_overlay(note="E", octave=4)) == "E4"
    assert overlay_key_color(_overlay(key_type="LB")) == "B"
    assert overlay_key_color(_overlay(key_type="RW")) == "W"


def test_unlit_frame_guard_returns_clean_for_uniform_keyboard_groups():
    frame = np.zeros((20, 80, 3), dtype=np.uint8)
    overlays = []
    for i in range(4):
        overlays.append(_overlay(key_id=i, note="C", octave=4, x=i * 10, y=0, width=8, height=8, key_type="LW"))
        frame[0:8, i * 10:i * 10 + 8] = (245, 245, 235)
    for i in range(4):
        overlays.append(_overlay(key_id=10 + i, note="C♯", octave=4, x=i * 10, y=10, width=8, height=8, key_type="LB"))
        frame[10:18, i * 10:i * 10 + 8] = (25, 25, 25)

    assessment = assess_unlit_frame(frame, overlays)

    assert assessment.status == "clean"
    assert assessment.likely_lit == ()
    assert assessment.reason == ""


def test_unlit_frame_guard_warns_with_likely_lit_note_name():
    frame = np.zeros((20, 80, 3), dtype=np.uint8)
    overlays = []
    for i in range(6):
        overlays.append(_overlay(key_id=i, note="E", octave=4, x=i * 10, y=0, width=8, height=8, key_type="LW"))
        frame[0:8, i * 10:i * 10 + 8] = (245, 245, 235)
    overlays[2].note_name_in_octave = "G"
    frame[0:8, 20:28] = (235, 150, 40)

    assessment = assess_unlit_frame(frame, overlays)

    assert assessment.status == "warning"
    assert assessment.reason == "color_outlier"
    assert [item.note_label for item in assessment.likely_lit] == ["G4"]
    assert assessment.likely_lit[0].confidence > 0.5


def test_unlit_frame_guard_reason_code_for_insufficient_samples():
    frame = np.zeros((20, 80, 3), dtype=np.uint8)
    overlays = [_overlay(key_id=1, note="C", octave=4, x=0, y=0, width=8, height=8, key_type="LW")]

    assessment = assess_unlit_frame(frame, overlays)

    assert assessment.status == "unknown"
    assert assessment.reason == "insufficient_samples"


def test_unlit_frame_reference_saturation_threshold_is_configurable():
    frame = np.zeros((20, 40, 3), dtype=np.uint8)
    overlays = []
    for i in range(4):
        overlay = _overlay(key_id=i, note="C", octave=4, x=i * 10, y=0, width=8, height=8, key_type="LW")
        overlays.append(overlay)
        frame[0:8, i * 10:i * 10 + 8] = (100, 255 if i == 3 else 240, 100)
    overlays[3].unlit_reference_color = (220, 0, 220)

    assert assess_unlit_frame(frame, overlays).status == "warning"
    assert assess_unlit_frame(frame, overlays, min_reference_saturation=255.0).status == "clean"


def test_capture_unlit_references_sets_rgb_and_histogram():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[1:5, 1:5] = (100, 120, 140)
    overlay = _overlay(x=1, y=1, width=4, height=4)

    count = capture_unlit_references_from_frame(frame, [overlay])

    assert count == 1
    assert overlay.unlit_reference_color == (100, 120, 140)
    assert overlay.unlit_hist is not None


def test_scanner_finds_lit_candidates_from_overlay_deltas():
    overlays = [
        _overlay(key_id=1, note="C", octave=4, x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C♯", octave=4, x=5, y=0, width=4, height=4, key_type="LB"),
    ]
    overlays[0].unlit_reference_color = (245, 245, 235)
    overlays[1].unlit_reference_color = (25, 25, 25)

    frames = {}
    for index in range(0, 31):
        frame = np.zeros((8, 16, 3), dtype=np.uint8)
        frame[:, :] = (10, 10, 10)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frames[index] = frame
    frames[20][0:4, 0:4] = (130, 165, 205)
    frames[21][0:4, 5:9] = (70, 110, 170)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda index: frames.get(index),
        overlays,
        0,
        30,
        settings=ExemplarScanSettings(coarse_stride=10, refine_radius=2, min_rgb_delta=30.0),
    )

    assert canceled is False
    assert scanned > 0
    assert {candidate.note_label for candidate in candidates} >= {"C4", "C♯4"}
    assert any(candidate.slot_color == "W" and candidate.rgb == (130, 165, 205) for candidate in candidates)
    assert any(candidate.slot_color == "B" and candidate.rgb == (70, 110, 170) for candidate in candidates)


def test_scanner_honors_cancel_callback():
    overlay = _overlay()
    overlay.unlit_reference_color = (245, 245, 235)
    frame = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda _index: frame,
        [overlay],
        0,
        100,
        progress_callback=lambda _current, _end: False,
    )

    assert candidates == []
    assert scanned == 0
    assert canceled is True
