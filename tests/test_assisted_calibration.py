import pytest

import cv2
import numpy as np

import synthesia2midi.detection.assisted_calibration as assisted_calibration
import synthesia2midi.detection.color_family_assignment as color_family_assignment
from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.assisted_calibration import (
    AssistedCalibrationProposal,
    ExemplarScanDiagnostics,
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


def _stable_candidates(
    slot_color,
    rgb,
    *,
    first_frame=10,
    note="C4",
    key_id=1,
    confidence=0.9,
):
    return [
        _candidate(
            slot_color,
            rgb,
            frame_index=first_frame,
            note=note,
            key_id=key_id,
            confidence=confidence,
        ),
        _candidate(
            slot_color,
            rgb,
            frame_index=first_frame + 10,
            note=note,
            key_id=key_id,
            confidence=confidence,
        ),
    ]


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


def _build_proposal_from_candidates(monkeypatch, candidates, *, saved_anchors=None):
    monkeypatch.setattr(
        assisted_calibration,
        "scan_lit_exemplar_candidates",
        lambda *_args, **_kwargs: (candidates, len(candidates), False),
    )
    return build_assisted_calibration_proposal(
        lambda _index: np.zeros((4, 4, 3), dtype=np.uint8),
        [],
        baseline_frame_index=0,
        end_frame=100,
        saved_anchors=saved_anchors,
    )


_SCANNER_FAMILY_COLORS = (
    ((70, 130, 230), (45, 95, 185)),
    ((235, 65, 65), (185, 35, 35)),
    ((235, 215, 45), (185, 165, 25)),
    ((45, 210, 70), (25, 160, 50)),
)

_WEAK_SCANNER_FAMILY_COLORS = (
    ((245, 210, 210), (60, 50, 50)),
    ((210, 245, 210), (50, 60, 50)),
    ((210, 210, 245), (50, 50, 60)),
    ((245, 210, 245), (60, 50, 60)),
)


def _four_family_scanner_fixture(events, *, colors=_SCANNER_FAMILY_COLORS):
    overlays = []
    for family_index in range(4):
        for morphology_index, key_color in enumerate(("W", "B")):
            overlay_index = (family_index * 2) + morphology_index
            overlay = _overlay(
                key_id=overlay_index + 1,
                note="C" if key_color == "W" else "C#",
                x=overlay_index * 5,
                y=0,
                width=4,
                height=4,
                key_type=f"L{key_color}",
            )
            overlay.unlit_reference_color = (
                (245, 245, 235) if key_color == "W" else (25, 25, 25)
            )
            overlays.append(overlay)

    def frame_provider(frame_index):
        frame = np.zeros((6, 40, 3), dtype=np.uint8)
        for overlay in overlays:
            x1 = int(overlay.x)
            x2 = x1 + int(overlay.width)
            frame[0:4, x1:x2] = overlay.unlit_reference_color
        for event in events:
            family_index, start_frame, end_frame = event[:3]
            morphology_indices = range(2) if len(event) == 3 else (event[3],)
            if start_frame <= frame_index <= end_frame:
                for morphology_index in morphology_indices:
                    overlay_index = (family_index * 2) + morphology_index
                    overlay = overlays[overlay_index]
                    x1 = int(overlay.x)
                    x2 = x1 + int(overlay.width)
                    frame[0:4, x1:x2] = colors[family_index][morphology_index]
        return frame

    return overlays, frame_provider


def _two_bursts(family_index, first_frame):
    return (
        (family_index, first_frame, first_frame + 10),
        (family_index, first_frame + 30, first_frame + 40),
    )


def _four_family_bursts(first_frame):
    return tuple(
        event
        for family_index in range(4)
        for event in _two_bursts(family_index, first_frame)
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


def test_scanner_continues_after_two_complete_families_and_finds_third_at_frame_900():
    events = (
        *_two_bursts(0, 100),
        *_two_bursts(1, 200),
        *_two_bursts(2, 900),
    )
    overlays, frame_provider = _four_family_scanner_fixture(events)
    diagnostics = ExemplarScanDiagnostics()

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1000,
        diagnostics=diagnostics,
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert scanned == 101
    assert assignment.family_count == 3
    assert any(candidate.frame_index == 900 for candidate in candidates)


def test_scanner_quiesces_refinement_during_long_two_family_video():
    events = (*_two_bursts(0, 100), *_two_bursts(1, 200))
    overlays, frame_provider = _four_family_scanner_fixture(events)
    diagnostics = ExemplarScanDiagnostics()

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1200,
        diagnostics=diagnostics,
    )

    assert canceled is False
    assert scanned == diagnostics.discovery_frames == 121
    assert assign_exemplar_slots(candidates).family_count == 2
    assert diagnostics.refined_frames < diagnostics.discovery_frames
    assert diagnostics.refined_events <= 4


def test_scanner_reactivates_refinement_for_late_third_family():
    events = (
        *_two_bursts(0, 100),
        *_two_bursts(1, 200),
        *_two_bursts(2, 900),
    )
    overlays, frame_provider = _four_family_scanner_fixture(events)
    diagnostics = ExemplarScanDiagnostics()

    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1200,
        diagnostics=diagnostics,
    )

    assert canceled is False
    assert assign_exemplar_slots(candidates).family_count == 3
    assert diagnostics.refined_events == 6


def test_scanner_does_not_early_stop_for_weak_four_family_evidence():
    overlays, frame_provider = _four_family_scanner_fixture(
        _four_family_bursts(100),
        colors=_WEAK_SCANNER_FAMILY_COLORS,
    )

    _candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1000,
    )

    assert canceled is False
    assert scanned == 101


def test_scanner_waits_for_confirmation_after_repeated_four_family_animation_pulses():
    overlays, frame_provider = _four_family_scanner_fixture(_four_family_bursts(100))

    _candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1000,
    )

    assert canceled is False
    assert scanned == 101


def test_scanner_stops_early_after_four_family_guards_pass():
    events = (
        *_four_family_bursts(100),
        *((family_index, 220, 230) for family_index in range(4)),
    )
    overlays, frame_provider = _four_family_scanner_fixture(events)
    diagnostics = ExemplarScanDiagnostics()

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        1000,
        diagnostics=diagnostics,
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert assignment.family_count == 4
    assert all(
        assignment.assignments[slot].hist is not None
        for slot in (
            "LW",
            "LB",
            "RW",
            "RB",
            "COLOR_3_W",
            "COLOR_3_B",
            "COLOR_4_W",
            "COLOR_4_B",
        )
    )
    assert scanned == diagnostics.discovery_frames < 101
    assert scanned >= 25


def test_scanner_rejects_one_frame_intro_flash_as_unstable():
    overlays, frame_provider = _four_family_scanner_fixture(
        tuple((family_index, 0, 0) for family_index in range(4))
    )
    diagnostics = ExemplarScanDiagnostics()

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        100,
        diagnostics=diagnostics,
    )

    assert canceled is False
    assert scanned == 11
    assert assign_exemplar_slots(candidates).family_count == 0
    assert diagnostics.refined_events == 0


def test_scanner_requires_stability_for_each_family_morphology():
    events = (
        (0, 20, 30, 0),
        (0, 60, 70, 0),
        (0, 30, 40, 1),
        (0, 70, 80, 1),
        (1, 40, 50, 0),
        (1, 80, 90, 0),
        (1, 50, 60, 1),
        (1, 90, 100, 1),
        (2, 100, 110, 0),
        (2, 140, 150, 0),
        (2, 110, 120, 1),
        (2, 150, 160, 1),
        (3, 160, 170, 0),
        (3, 200, 210, 0),
        (3, 170, 180, 1),
    )
    overlays, frame_provider = _four_family_scanner_fixture(events)
    diagnostics = ExemplarScanDiagnostics()

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        300,
        diagnostics=diagnostics,
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert scanned == diagnostics.discovery_frames == 31
    assert assignment.family_count == 4
    assert assignment.missing_slots == ("COLOR_4_B",)
    assert assignment.assignments["COLOR_4_B"].hist is None
    assert diagnostics.refined_events == 7


def test_scanner_rerefines_only_a_materially_stronger_stable_candidate():
    overlays, base_provider = _four_family_scanner_fixture(())
    overlays = overlays[:2]
    weak_natural = (170, 195, 215)
    strong_natural = (140, 170, 210)
    nearby_natural = (135, 168, 210)
    accidental = _SCANNER_FAMILY_COLORS[0][1]

    def frame_provider(frame_index):
        frame = base_provider(frame_index)
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 0:4] = weak_natural
        elif 300 <= frame_index <= 310:
            frame[0:4, 0:4] = strong_natural
        elif 400 <= frame_index <= 410:
            frame[0:4, 0:4] = nearby_natural
        elif 500 <= frame_index <= 510:
            frame[0:4, 0:4] = weak_natural
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 5:9] = accidental
        return frame

    diagnostics = ExemplarScanDiagnostics()
    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        550,
        diagnostics=diagnostics,
    )

    assignment = assign_exemplar_slots(candidates)
    selected = assignment.assignments["LW"]
    assert canceled is False
    assert diagnostics.refined_events == 3
    assert selected.rgb == strong_natural
    assert selected.source is not None
    assert selected.source.frame_index == 300
    assert selected.hist is not None


def test_scanner_rerefines_stronger_candidate_after_confidence_saturates():
    overlays, base_provider = _four_family_scanner_fixture(())
    overlays = overlays[:2]
    initial_natural = (80, 120, 200)
    stronger_natural = (20, 60, 180)
    accidental = _SCANNER_FAMILY_COLORS[0][1]

    def frame_provider(frame_index):
        frame = base_provider(frame_index)
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 0:4] = initial_natural
        elif 300 <= frame_index <= 310:
            frame[0:4, 0:4] = stronger_natural
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 5:9] = accidental
        return frame

    diagnostics = ExemplarScanDiagnostics()
    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        350,
        diagnostics=diagnostics,
    )

    selected = assign_exemplar_slots(candidates).assignments["LW"]
    assert canceled is False
    assert diagnostics.refined_events == 3
    assert selected.rgb == stronger_natural
    assert selected.source is not None
    assert selected.source.frame_index == 300
    assert selected.hist is not None


def test_scanner_retains_refined_exemplar_after_quiescent_events_fill_bound():
    overlays, base_provider = _four_family_scanner_fixture(())
    overlays = overlays[:2]
    refined_natural = (140, 170, 210)
    nearby_naturals = (
        (138, 169, 210),
        (136, 168, 210),
        (134, 167, 210),
    )
    accidental = _SCANNER_FAMILY_COLORS[0][1]

    def frame_provider(frame_index):
        frame = base_provider(frame_index)
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 0:4] = refined_natural
        for event_index, rgb in zip((300, 400, 500), nearby_naturals):
            if event_index <= frame_index <= event_index + 10:
                frame[0:4, 0:4] = rgb
        if 100 <= frame_index <= 110 or 140 <= frame_index <= 150:
            frame[0:4, 5:9] = accidental
        return frame

    diagnostics = ExemplarScanDiagnostics()
    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        550,
        diagnostics=diagnostics,
    )

    selected = assign_exemplar_slots(candidates).assignments["LW"]
    assert canceled is False
    assert diagnostics.refined_events == 2
    assert selected.rgb == refined_natural
    assert selected.source is not None
    assert selected.source.frame_index == 100
    assert selected.hist is not None


def test_scanner_clustering_work_scales_near_linearly_with_event_count(monkeypatch):
    work_counts = []
    assignment_hue_calls = []
    hue_call_count = 0
    original_hue_distance = color_family_assignment._circular_hue_distance

    def count_hue_distance(left, right):
        nonlocal hue_call_count
        hue_call_count += 1
        return original_hue_distance(left, right)

    monkeypatch.setattr(
        color_family_assignment,
        "_circular_hue_distance",
        count_hue_distance,
    )
    for event_count in (10, 20, 40):
        hue_calls_before_scan = hue_call_count
        overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4, key_type="LW")
        overlay.unlit_reference_color = (245, 245, 235)

        def frame_provider(frame_index, *, event_count=event_count):
            frame = np.full((6, 6, 3), (245, 245, 235), dtype=np.uint8)
            if frame_index % 20 == 0 and frame_index // 20 < event_count:
                frame[0:4, 0:4] = _SCANNER_FAMILY_COLORS[0][0]
            return frame

        diagnostics = ExemplarScanDiagnostics()
        candidates, _scanned, canceled = scan_lit_exemplar_candidates(
            frame_provider,
            [overlay],
            0,
            event_count * 20,
            diagnostics=diagnostics,
        )

        assert canceled is False
        assignment_hue_calls.append(hue_call_count - hue_calls_before_scan)
        assert assign_exemplar_slots(candidates).family_count == 1
        assert diagnostics.max_clustering_evidence <= 3
        work_counts.append(diagnostics.clustering_work)

    assert work_counts == [34, 74, 154]
    assert assignment_hue_calls == [33, 73, 153]


def test_scanner_collapses_sustained_note_into_one_discovery_event():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    def frame_provider(frame_index):
        frame = np.full((6, 6, 3), (245, 245, 235), dtype=np.uint8)
        if 100 <= frame_index <= 200:
            frame[0:4, 0:4] = (70, 130, 230)
        return frame

    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        [overlay],
        0,
        300,
    )

    assert canceled is False
    assert len(candidates) == 1
    assert candidates[0].frame_index == 100


def test_scanner_merges_nearby_hues_into_one_stable_family():
    overlays, base_provider = _four_family_scanner_fixture(())
    overlays = overlays[:2]

    def frame_provider(frame_index):
        frame = base_provider(frame_index)
        burst_colors = None
        if 100 <= frame_index <= 110:
            burst_colors = ((70, 130, 230), (45, 95, 185))
        elif 140 <= frame_index <= 150:
            burst_colors = ((75, 145, 225), (50, 105, 180))
        if burst_colors is not None:
            for overlay, rgb in zip(overlays, burst_colors):
                x1 = int(overlay.x)
                x2 = x1 + int(overlay.width)
                frame[0:4, x1:x2] = rgb
        return frame

    candidates, _scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        200,
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert assignment.family_count == 1
    assert assignment.missing_slots == ()


def test_scanner_discovers_event_on_ten_frame_checkpoint():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    def frame_provider(frame_index):
        frame = np.full((6, 6, 3), (245, 245, 235), dtype=np.uint8)
        if frame_index == 20:
            frame[0:4, 0:4] = (70, 130, 230)
        return frame

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        [overlay],
        0,
        30,
    )

    assert canceled is False
    assert scanned == 4
    assert [candidate.frame_index for candidate in candidates] == [20]


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
    frames[20][0:4, 5:9] = (70, 110, 170)

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
    assert set(by_frame) == {12, 25}
    assert by_frame[12].rgb == (50, 70, 90)
    assert by_frame[25].rgb == (175, 195, 215)


def test_scanner_bounds_completed_candidates_while_scanning(monkeypatch):
    overlay = _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    frames = {}
    for frame_index in range(31):
        frame = np.full((6, 6, 3), (245, 245, 235), dtype=np.uint8)
        if frame_index % 2 == 1:
            frame[0:4, 0:4] = (130, 165, 205)
        frames[frame_index] = frame

    largest_completed_bucket = 0
    original_flatten = assisted_calibration._flatten_scan_candidates

    def record_bucket_size(candidates_by_key, active_candidates_by_key, max_candidates_per_key):
        nonlocal largest_completed_bucket
        largest_completed_bucket = max(
            [largest_completed_bucket]
            + [len(bucket) for bucket in candidates_by_key.values()]
        )
        return original_flatten(
            candidates_by_key,
            active_candidates_by_key,
            max_candidates_per_key,
        )

    monkeypatch.setattr(
        assisted_calibration,
        "_flatten_scan_candidates",
        record_bucket_size,
    )
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda index: frames[index],
        [overlay],
        0,
        30,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
            max_candidates_per_key=3,
        ),
    )

    assert canceled is False
    assert scanned == 31
    assert largest_completed_bucket <= 3
    assert [candidate.frame_index for candidate in candidates] == [1, 3, 5]


def test_scanner_does_not_refine_single_completed_burst():
    overlay = _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW")
    overlay.unlit_reference_color = (245, 245, 235)

    frames = {}
    for frame_index in range(21):
        frame = np.full((6, 6, 3), (245, 245, 235), dtype=np.uint8)
        if frame_index == 1:
            frame[0:4, 0:4] = (130, 165, 205)
        frames[frame_index] = frame

    diagnostics = ExemplarScanDiagnostics()
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        lambda index: frames[index],
        [overlay],
        0,
        20,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
        diagnostics=diagnostics,
    )

    assert canceled is False
    assert scanned == 21
    assert len(candidates) == 1
    assert candidates[0].hist is None
    assert diagnostics.refined_events == 0


def test_scanner_skips_refinement_when_discovery_evidence_is_not_stable():
    overlay = _overlay(
        key_id=1, note="C", octave=4, x=0, y=0, width=4, height=4, key_type="LW"
    )
    overlay.unlit_reference_color = (245, 245, 235)

    frames = {}
    for index in range(0, 6):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :] = (25, 25, 25)
        frame[0:4, 0:4] = (245, 245, 235)
        frames[index] = frame
    frames[0][0:4, 0:4] = (130, 165, 205)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames.get(index)

    diagnostics = ExemplarScanDiagnostics()
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        [overlay],
        0,
        5,
        settings=ExemplarScanSettings(
            coarse_stride=5, refine_radius=1, min_rgb_delta=30.0
        ),
        diagnostics=diagnostics,
    )

    assert canceled is False
    assert scanned == 2
    assert len(candidates) == 1
    assert candidates[0].frame_index == 0
    assert diagnostics.refined_frames == 0
    assert calls == [0, 5]


def test_scanner_continues_to_end_after_confirmed_two_family_exemplars():
    overlays = [
        _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C#", x=5, y=0, width=4, height=4, key_type="LB"),
        _overlay(key_id=3, note="D", x=10, y=0, width=4, height=4, key_type="RW"),
        _overlay(key_id=4, note="D#", x=15, y=0, width=4, height=4, key_type="RB"),
    ]
    for overlay in overlays:
        overlay.unlit_reference_color = (
            (25, 25, 25) if overlay.key_type.endswith("B") else (245, 245, 235)
        )

    frames = {}
    for index in range(41):
        frame = np.zeros((6, 20, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frame[0:4, 10:14] = (245, 245, 235)
        frame[0:4, 15:19] = (25, 25, 25)
        frames[index] = frame

    lit_events = {
        1: (0, (130, 165, 205)),
        2: (1, (70, 110, 170)),
        3: (2, (243, 176, 68)),
        4: (3, (243, 131, 46)),
        9: (0, (130, 165, 205)),
        10: (1, (70, 110, 170)),
        11: (2, (243, 176, 68)),
        12: (3, (243, 131, 46)),
        15: (0, (130, 165, 205)),
        16: (1, (70, 110, 170)),
        17: (2, (243, 176, 68)),
        18: (3, (243, 131, 46)),
    }
    for frame_index, (overlay_index, rgb) in lit_events.items():
        overlay = overlays[overlay_index]
        frames[frame_index][0:4, int(overlay.x):int(overlay.x + overlay.width)] = rgb

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    diagnostics = ExemplarScanDiagnostics()
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        40,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
        diagnostics=diagnostics,
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert assignment.family_count == 2
    assert assignment.missing_slots == ()
    assert scanned == 41
    assert diagnostics.refined_events == 4
    assert set(range(41)).issubset(calls)


def test_scanner_uses_fresh_evidence_after_output_candidate_bucket_is_full():
    overlays = [
        _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C#", x=5, y=0, width=4, height=4, key_type="LB"),
        _overlay(key_id=3, note="D", x=10, y=0, width=4, height=4, key_type="RW"),
        _overlay(key_id=4, note="D#", x=15, y=0, width=4, height=4, key_type="RB"),
    ]
    for overlay in overlays:
        overlay.unlit_reference_color = (
            (25, 25, 25) if overlay.key_type.endswith("B") else (245, 245, 235)
        )

    frames = {}
    for frame_index in range(41):
        frame = np.zeros((6, 20, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frame[0:4, 10:14] = (245, 245, 235)
        frame[0:4, 15:19] = (25, 25, 25)
        frames[frame_index] = frame

    for frame_index in (1, 3, 5, 7, 9, 11, 13):
        frames[frame_index][0:4, 0:4] = (130, 165, 205)
        frames[frame_index][0:4, 5:9] = (70, 110, 170)
        frames[frame_index][0:4, 10:14] = (243, 176, 68)
    for frame_index in (15, 17):
        frames[frame_index][0:4, 15:19] = (243, 131, 46)
    frames[19][0:4, 0:4] = (130, 165, 205)
    frames[19][0:4, 5:9] = (70, 110, 170)
    frames[19][0:4, 10:14] = (243, 176, 68)
    frames[19][0:4, 15:19] = (243, 131, 46)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        40,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
            max_candidates_per_key=6,
        ),
    )

    assert canceled is False
    assert assign_exemplar_slots(candidates).missing_slots == ()
    assert scanned == 41
    assert set(range(41)).issubset(calls)


def test_scanner_does_not_treat_reordered_color_families_as_fresh_evidence():
    overlays = [
        _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C#", x=5, y=0, width=4, height=4, key_type="LB"),
        _overlay(key_id=3, note="D", x=10, y=0, width=4, height=4, key_type="RW"),
        _overlay(key_id=4, note="D#", x=15, y=0, width=4, height=4, key_type="RB"),
    ]
    for overlay in overlays:
        overlay.unlit_reference_color = (
            (25, 25, 25) if overlay.key_type.endswith("B") else (245, 245, 235)
        )

    frames = {}
    for frame_index in range(31):
        frame = np.zeros((6, 20, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frame[0:4, 10:14] = (245, 245, 235)
        frame[0:4, 15:19] = (25, 25, 25)
        frames[frame_index] = frame

    for frame_index in (1, 5, 9, 11):
        frames[frame_index][0:4, 0:4] = (230, 80, 80)
        frames[frame_index][0:4, 5:9] = (180, 40, 40)
    for frame_index in (3, 7):
        frames[frame_index][0:4, 10:14] = (230, 220, 80)
        frames[frame_index][0:4, 15:19] = (180, 160, 40)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        30,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
    )

    assert canceled is False
    assert assign_exemplar_slots(candidates).family_count == 2
    assert scanned == 31
    assert set(range(31)).issubset(calls)


def test_scanner_does_not_stop_for_two_transient_animation_pulses():
    overlays = []
    for index in range(8):
        key_color = "B" if index in {2, 3, 6, 7} else "W"
        overlay = _overlay(
            key_id=index + 1,
            note="C" if key_color == "W" else "C#",
            x=index * 5,
            y=0,
            width=4,
            height=4,
            key_type=f"L{key_color}",
        )
        overlay.unlit_reference_color = (
            (25, 25, 25) if key_color == "B" else (245, 245, 235)
        )
        overlays.append(overlay)

    frames = {}
    for frame_index in range(21):
        frame = np.zeros((6, 40, 3), dtype=np.uint8)
        for overlay in overlays:
            unlit_rgb = overlay.unlit_reference_color
            frame[0:4, int(overlay.x):int(overlay.x + overlay.width)] = unlit_rgb
        frames[frame_index] = frame

    animation_colors = [
        (130, 165, 205),
        (130, 165, 205),
        (70, 110, 170),
        (70, 110, 170),
        (243, 176, 68),
        (243, 176, 68),
        (243, 131, 46),
        (243, 131, 46),
    ]
    for frame_index in (1, 3):
        for overlay, rgb in zip(overlays, animation_colors):
            frames[frame_index][0:4, int(overlay.x):int(overlay.x + overlay.width)] = rgb

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    _candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        20,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
    )

    assert canceled is False
    assert scanned == 21
    assert max(calls) == 20


def test_scanner_scans_to_end_for_one_complete_color_family():
    overlays = [
        _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C#", x=5, y=0, width=4, height=4, key_type="LB"),
    ]
    overlays[0].unlit_reference_color = (245, 245, 235)
    overlays[1].unlit_reference_color = (25, 25, 25)

    frames = {}
    for frame_index in range(21):
        frame = np.zeros((6, 10, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frames[frame_index] = frame
    for frame_index in (1, 9):
        frames[frame_index][0:4, 0:4] = (130, 165, 205)
    for frame_index in (2, 10):
        frames[frame_index][0:4, 5:9] = (70, 110, 170)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        20,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
    )

    assert canceled is False
    assert assign_exemplar_slots(candidates).family_count == 1
    assert scanned == 21
    assert set(range(21)).issubset(calls)


def test_scanner_scans_to_end_when_second_family_is_incomplete():
    overlays = [
        _overlay(key_id=1, note="C", x=0, y=0, width=4, height=4, key_type="LW"),
        _overlay(key_id=2, note="C#", x=5, y=0, width=4, height=4, key_type="LB"),
        _overlay(key_id=3, note="D", x=10, y=0, width=4, height=4, key_type="RW"),
    ]
    for overlay in overlays:
        overlay.unlit_reference_color = (
            (25, 25, 25) if overlay.key_type.endswith("B") else (245, 245, 235)
        )

    frames = {}
    for frame_index in range(21):
        frame = np.zeros((6, 15, 3), dtype=np.uint8)
        frame[0:4, 0:4] = (245, 245, 235)
        frame[0:4, 5:9] = (25, 25, 25)
        frame[0:4, 10:14] = (245, 245, 235)
        frames[frame_index] = frame
    for frame_index in (1, 9):
        frames[frame_index][0:4, 0:4] = (130, 165, 205)
    for frame_index in (2, 10):
        frames[frame_index][0:4, 5:9] = (70, 110, 170)
    for frame_index in (3, 11):
        frames[frame_index][0:4, 10:14] = (243, 176, 68)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return frames[index]

    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        0,
        20,
        settings=ExemplarScanSettings(
            coarse_stride=1,
            refine_radius=0,
            min_rgb_delta=30.0,
        ),
    )

    assignment = assign_exemplar_slots(candidates)
    assert canceled is False
    assert assignment.family_count == 2
    assert assignment.missing_slots == ("RB",)
    assert scanned == 21
    assert set(range(21)).issubset(calls)


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
        *_stable_candidates("W", (130, 165, 205), key_id=50, note="D5"),
        *_stable_candidates("B", (70, 110, 170), key_id=30, note="C♯4"),
        *_stable_candidates("W", (243, 176, 68), key_id=10, note="A2"),
        *_stable_candidates("B", (243, 131, 46), key_id=20, note="A♯2"),
    ]

    result = assign_exemplar_slots(candidates)

    assert result.family_count == 2
    assert result.assignments["LW"].rgb == (243, 176, 68)
    assert result.assignments["LB"].rgb == (243, 131, 46)
    assert result.assignments["RW"].rgb == (130, 165, 205)
    assert result.assignments["RB"].rgb == (70, 110, 170)
    assert result.disabled_slots == (
        "COLOR_3_W",
        "COLOR_3_B",
        "COLOR_4_W",
        "COLOR_4_B",
    )


def test_assign_exemplar_slots_disables_absent_second_family():
    result = assign_exemplar_slots([
        *_stable_candidates("W", (130, 165, 205), key_id=1),
        *_stable_candidates("B", (70, 110, 170), key_id=2),
    ])

    assert result.family_count == 1
    assert result.assignments["LW"].enabled is True
    assert result.assignments["LB"].enabled is True
    assert result.assignments["RW"].enabled is False
    assert result.assignments["RB"].enabled is False
    assert result.disabled_slots == (
        "RW",
        "RB",
        "COLOR_3_W",
        "COLOR_3_B",
        "COLOR_4_W",
        "COLOR_4_B",
    )


def test_assign_exemplar_slots_enables_partial_family_with_missing_partner_as_missing():
    result = assign_exemplar_slots([
        *_stable_candidates("W", (130, 165, 205), key_id=1),
    ])

    assert result.family_count == 1
    assert result.assignments["LW"].enabled is True
    assert result.assignments["LW"].rgb == (130, 165, 205)
    assert result.assignments["LW"].hist is not None
    assert result.assignments["LB"].enabled is True
    assert result.assignments["LB"].rgb is None
    assert result.assignments["LB"].hist is None
    assert result.missing_slots == ("LB",)
    assert result.disabled_slots == (
        "RW",
        "RB",
        "COLOR_3_W",
        "COLOR_3_B",
        "COLOR_4_W",
        "COLOR_4_B",
    )


def test_assign_exemplar_slots_maps_four_stable_families_to_registry_slots():
    family_samples = (
        ((70, 130, 230), (45, 95, 185)),
        ((235, 65, 65), (185, 35, 35)),
        ((235, 215, 45), (185, 165, 25)),
        ((45, 210, 70), (25, 160, 50)),
    )
    candidates = []
    for family_index, (natural_rgb, accidental_rgb) in enumerate(family_samples):
        candidates.extend(
            _stable_candidates(
                "W",
                natural_rgb,
                first_frame=10 + (family_index * 30),
                key_id=10 + family_index,
            )
        )
        candidates.extend(
            _stable_candidates(
                "B",
                accidental_rgb,
                first_frame=11 + (family_index * 30),
                key_id=20 + family_index,
            )
        )

    result = assign_exemplar_slots(list(reversed(candidates)))

    assert result.family_count == 4
    assert tuple(result.assignments) == (
        "LW",
        "LB",
        "RW",
        "RB",
        "COLOR_3_W",
        "COLOR_3_B",
        "COLOR_4_W",
        "COLOR_4_B",
    )
    assert result.missing_slots == ()
    assert result.disabled_slots == ()
    assert all(assignment.enabled for assignment in result.assignments.values())


def test_assign_exemplar_slots_selects_duplicate_source_deterministically():
    duplicate_a = _candidate("W", (130, 165, 205), frame_index=10, key_id=1)
    duplicate_b = ExemplarCandidate(
        slot_color=duplicate_a.slot_color,
        key_id=duplicate_a.key_id,
        note_label=duplicate_a.note_label,
        frame_index=duplicate_a.frame_index,
        rgb=duplicate_a.rgb,
        hsv=duplicate_a.hsv,
        delta_from_unlit=duplicate_a.delta_from_unlit,
        confidence=duplicate_a.confidence,
        hist=np.array([2.0], dtype=np.float32),
    )
    later = _candidate("W", (130, 165, 205), frame_index=20, key_id=1)

    forward = assign_exemplar_slots([duplicate_a, duplicate_b, later])
    reversed_result = assign_exemplar_slots([later, duplicate_b, duplicate_a])

    assert np.array_equal(
        forward.assignments["LW"].hist,
        reversed_result.assignments["LW"].hist,
    )


def test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots():
    app_state = AppState()
    assignment = assign_exemplar_slots([
        *_stable_candidates("W", (130, 165, 205), key_id=1),
        *_stable_candidates("B", (70, 110, 170), key_id=2),
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


def test_assisted_calibration_proposal_preserves_positional_canceled_argument():
    proposal = AssistedCalibrationProposal(
        12,
        UnlitFrameAssessment(status="clean"),
        assign_exemplar_slots([]),
        3,
        0,
        True,
    )

    assert proposal.canceled is True
    assert proposal.warnings == ()


def test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4)
    baseline = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)
    lit = baseline.copy()
    lit[0:4, 0:4] = (130, 165, 205)
    frames = {0: baseline, 5: lit, 10: lit}

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
    assert proposal.candidate_count == 2
    assert proposal.assignment_result.assignments["LW"].rgb == (130, 165, 205)


def test_build_assisted_calibration_proposal_skips_baseline_frame_as_lit_candidate():
    overlay = _overlay(key_id=1, x=0, y=0, width=4, height=4)
    overlay.unlit_reference_color = (245, 245, 235)

    baseline = np.full((8, 8, 3), (170, 175, 185), dtype=np.uint8)
    unlit = np.full((8, 8, 3), (245, 245, 235), dtype=np.uint8)
    lit = np.full((8, 8, 3), (150, 155, 165), dtype=np.uint8)
    lit[0:4, 0:4] = (130, 165, 205)

    calls: list[int] = []

    def frame_provider(index: int):
        calls.append(index)
        return {0: baseline, 1: lit, 2: unlit, 3: unlit, 4: lit}.get(index)

    proposal = build_assisted_calibration_proposal(
        frame_provider,
        [overlay],
        baseline_frame_index=0,
        end_frame=4,
        settings=ExemplarScanSettings(coarse_stride=1, refine_radius=0, min_rgb_delta=30.0),
    )

    assert calls.count(0) == 1
    assert proposal.scanned_frame_count == 4
    assert proposal.candidate_count == 2
    assert proposal.assignment_result.assignments["LW"].rgb == (130, 165, 205)


def test_build_assisted_calibration_proposal_carries_over_cap_warning(monkeypatch):
    family_samples = (
        ((70, 130, 230), (45, 95, 185)),
        ((235, 65, 65), (185, 35, 35)),
        ((235, 215, 45), (185, 165, 25)),
        ((45, 210, 70), (25, 160, 50)),
        ((165, 80, 220), (115, 45, 170)),
    )
    candidates = []
    for family_index, (natural_rgb, accidental_rgb) in enumerate(family_samples):
        candidates.extend(
            _stable_candidates(
                "W",
                natural_rgb,
                first_frame=10 + (family_index * 30),
                key_id=10 + family_index,
            )
        )
        candidates.extend(
            _stable_candidates(
                "B",
                accidental_rgb,
                first_frame=11 + (family_index * 30),
                key_id=20 + family_index,
            )
        )

    proposal = _build_proposal_from_candidates(monkeypatch, candidates)

    assert proposal.warnings == ("More than four stable color families were found.",)


def test_build_assisted_calibration_proposal_carries_anchor_conflict_warning(monkeypatch):
    natural_rgb = (70, 130, 230)
    accidental_rgb = (45, 95, 185)
    candidates = [
        *_stable_candidates("W", natural_rgb, key_id=1),
        *_stable_candidates("B", accidental_rgb, first_frame=11, key_id=2),
    ]

    proposal = _build_proposal_from_candidates(
        monkeypatch,
        candidates,
        saved_anchors={
            1: {"natural": natural_rgb},
            2: {"accidental": accidental_rgb},
        },
    )

    assert proposal.warnings == (
        "Evidence conflicts with two saved color family identities.",
    )


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
