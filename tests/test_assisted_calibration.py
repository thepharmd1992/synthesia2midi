import pytest

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.assisted_calibration import (
    AssistedCalibrationProposal,
    ExemplarScanSettings,
    ExemplarCandidate,
    _BoundedFrameCache,
    UnlitFrameAssessment,
    apply_assisted_calibration_proposal,
    assess_unlit_frame,
    assign_exemplar_slots,
    build_assisted_calibration_proposal,
    capture_unlit_references_from_frame,
    overlay_key_color,
    overlay_note_label,
    sample_overlay_bgr,
    sample_overlay_rgb,
    scan_lit_exemplar_candidates,
)
from synthesia2midi.tools.probe_assisted_calibration import _load_exemplar_lit_color_targets


def _candidate(slot_color, rgb, frame_index=10, note="C4", key_id=1, confidence=0.9):
    hsv = cv2.cvtColor(np.array([[rgb]], dtype=np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
    return ExemplarCandidate(
        slot_color=slot_color,
        key_id=key_id,
        note_label=note,
        frame_index=frame_index,
        rgb=rgb,
        hsv=(float(hsv[0]), float(hsv[1]), float(hsv[2])),
        delta_from_unlit=100.0,
        confidence=confidence,
        hist=np.array([1.0], dtype=np.float32),
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


def test_scanner_collapses_overlapping_hits_into_one_event_per_lit_burst():
    overlay = _overlay(key_id=1, note="C", octave=4, x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    frames = {}
    for index in range(0, 31):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :] = (25, 25, 25)
        frame[0:4, 0:4] = (245, 245, 235)
        frames[index] = frame

    frames[10][0:4, 0:4] = (170, 190, 210)
    frames[12][0:4, 0:4] = (50, 70, 90)
    frames[15][0:4, 0:4] = (180, 200, 220)
    frames[25][0:4, 0:4] = (175, 195, 215)
    frames[26][0:4, 0:4] = (45, 65, 85)
    frames[27][0:4, 0:4] = (185, 205, 225)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda index: frames.get(index),
        [overlay],
        0,
        30,
        settings=ExemplarScanSettings(coarse_stride=5, refine_radius=2, min_rgb_delta=30.0),
    )

    assert canceled is False
    assert scanned > 0
    assert len(candidates) == 2
    by_frame = {candidate.frame_index: candidate for candidate in candidates}
    assert set(by_frame) == {12, 26}
    assert by_frame[12].rgb == (50, 70, 90)
    assert by_frame[26].rgb == (45, 65, 85)


def test_scanner_reuses_frame_provider_reads_across_refinement_window():
    overlay = _overlay(key_id=1, note="C", octave=4, x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    frames = {}
    for index in range(0, 6):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :] = (25, 25, 25)
        frame[0:4, 0:4] = (245, 245, 235)
        frames[index] = frame
    frames[1][0:4, 0:4] = (130, 165, 205)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames.get(index)

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        [overlay],
        0,
        5,
        settings=ExemplarScanSettings(coarse_stride=5, refine_radius=1, min_rgb_delta=30.0),
    )

    assert canceled is False
    assert scanned == 2
    assert len(candidates) == 1
    assert candidates[0].frame_index == 1
    assert calls == [0, 1, 5, 4]


def test_bounded_frame_cache_evicts_old_frames():
    cache = _BoundedFrameCache(max_size=2)
    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return np.full((2, 2, 3), index, dtype=np.uint8)

    assert cache.get(0, frame_provider) is not None
    assert cache.get(1, frame_provider) is not None
    assert cache.get(0, frame_provider) is not None
    assert cache.get(2, frame_provider) is not None
    assert cache.get(1, frame_provider) is not None

    assert calls == [0, 1, 2, 1]
    assert len(cache) <= 2


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


def test_assign_exemplar_slots_maps_two_color_families_by_hue_not_position():
    candidates = [
        _candidate("W", (130, 165, 205), key_id=50, note="D5"),
        _candidate("B", (70, 110, 170), key_id=30, note="C♯4"),
        _candidate("W", (243, 176, 68), key_id=10, note="A2"),
        _candidate("B", (243, 131, 46), key_id=20, note="A♯2"),
    ]

    result = assign_exemplar_slots(candidates)

    assert result.family_count == 2
    assert result.assignments["LW"].rgb == (130, 165, 205)
    assert result.assignments["LB"].rgb == (70, 110, 170)
    assert result.assignments["RW"].rgb == (243, 176, 68)
    assert result.assignments["RB"].rgb == (243, 131, 46)
    assert result.disabled_slots == ()


def test_assign_exemplar_slots_disables_absent_second_family():
    result = assign_exemplar_slots([
        _candidate("W", (130, 165, 205), key_id=1),
        _candidate("B", (70, 110, 170), key_id=2),
    ])

    assert result.family_count == 1
    assert result.assignments["LW"].enabled is True
    assert result.assignments["LB"].enabled is True
    assert result.assignments["RW"].enabled is False
    assert result.assignments["RB"].enabled is False
    assert result.disabled_slots == ("RW", "RB")


def test_assign_exemplar_slots_enables_partial_family_with_missing_partner_as_missing():
    result = assign_exemplar_slots([
        _candidate("W", (130, 165, 205), key_id=1),
    ])

    assert result.family_count == 1
    assert result.assignments["LW"].enabled is True
    assert result.assignments["LW"].rgb == (130, 165, 205)
    assert result.assignments["LW"].hist is not None
    assert result.assignments["LB"].enabled is True
    assert result.assignments["LB"].rgb is None
    assert result.assignments["LB"].hist is None
    assert result.missing_slots == ("LB",)
    assert result.disabled_slots == ("RW", "RB")


def test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots():
    app_state = AppState()
    assignment = assign_exemplar_slots([
        _candidate("W", (130, 165, 205), key_id=1),
        _candidate("B", (70, 110, 170), key_id=2),
    ])
    proposal = AssistedCalibrationProposal(
        baseline_frame_index=12,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=assignment,
        scanned_frame_count=3,
        candidate_count=2,
    )

    apply_assisted_calibration_proposal(app_state, proposal)

    assert app_state.detection.exemplar_lit_colors["LW"] == (130, 165, 205)
    assert app_state.detection.exemplar_lit_colors["LB"] == (70, 110, 170)
    assert np.array_equal(app_state.detection.exemplar_lit_histograms["LW"], np.array([1.0], dtype=np.float32))
    assert np.array_equal(app_state.detection.exemplar_lit_histograms["LB"], np.array([1.0], dtype=np.float32))
    assert app_state.detection.exemplar_lit_colors["RW"] is None
    assert app_state.detection.exemplar_lit_colors["RB"] is None
    assert app_state.detection.exemplar_lit_histograms["RW"] is None
    assert app_state.detection.exemplar_lit_histograms["RB"] is None
    assert app_state.detection.exemplar_key_type_enabled["RW"] is False
    assert app_state.detection.exemplar_key_type_enabled["RB"] is False
    assert app_state.unsaved_changes is True


def test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4)
    baseline = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)
    lit = baseline.copy()
    lit[0:4, 0:4] = (130, 165, 205)
    frames = {0: baseline, 10: lit}

    capture_unlit_references_from_frame(baseline, [overlay])
    proposal = build_assisted_calibration_proposal(
        lambda index: frames.get(index, baseline),
        [overlay],
        baseline_frame_index=0,
        end_frame=10,
        settings=ExemplarScanSettings(coarse_stride=1, refine_radius=0, min_rgb_delta=30.0),
    )

    assert proposal.baseline_frame_index == 0
    assert proposal.unlit_assessment.status == "clean"
    assert proposal.candidate_count == 1
    assert proposal.assignment_result.assignments["LW"].rgb == (130, 165, 205)


def test_build_assisted_calibration_proposal_skips_baseline_frame_as_lit_candidate():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4)
    overlay.unlit_reference_color = (245, 245, 235)

    baseline = np.full((8, 8, 3), (170, 175, 185), dtype=np.uint8)
    lit = np.full((8, 8, 3), (150, 155, 165), dtype=np.uint8)
    lit[0:4, 0:4] = (130, 165, 205)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return {0: baseline, 1: lit}.get(index)

    proposal = build_assisted_calibration_proposal(
        frame_provider,
        [overlay],
        baseline_frame_index=0,
        end_frame=1,
        settings=ExemplarScanSettings(coarse_stride=1, refine_radius=0, min_rgb_delta=30.0),
    )

    assert calls.count(0) == 1
    assert proposal.scanned_frame_count == 1
    assert proposal.candidate_count == 1
    assert proposal.assignment_result.assignments["LW"].rgb == (130, 165, 205)


def test_load_exemplar_lit_color_targets_reads_configured_slots(tmp_path):
    ini_path = tmp_path / "sample.ini"
    ini_path.write_text(
        "\n".join(
            [
                "[ExemplarLitColors]",
                "lw = 130, 165, 205",
                "lb = ",
                "rw = 243, 176, 68",
                "rb = 243,131,46",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert _load_exemplar_lit_color_targets(ini_path) == {
        "LW": (130, 165, 205),
        "RW": (243, 176, 68),
        "RB": (243, 131, 46),
    }


def test_load_exemplar_lit_color_targets_requires_section(tmp_path):
    ini_path = tmp_path / "missing_section.ini"
    ini_path.write_text("[Other]\nvalue = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"ExemplarLitColors"):
        _load_exemplar_lit_color_targets(ini_path)
