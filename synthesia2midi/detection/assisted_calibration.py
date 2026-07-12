"""Pure data models and overlay sampling helpers for assisted calibration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.color_families import (
    COLOR_FAMILIES,
    SUPPORTED_EXEMPLAR_SLOTS,
    slots_for_family,
)
from synthesia2midi.detection.color_family_assignment import (
    FamilyEvidence,
    SavedFamilyAnchors,
    assign_family_slots,
)

if TYPE_CHECKING:
    from synthesia2midi.core.app_state import AppState

KeyColor = Literal["W", "B"]
AssessmentStatus = Literal["clean", "warning", "unknown"]
FrameProvider = Callable[[int], Optional[np.ndarray]]
ProgressCallback = Callable[[int, int], bool]


@dataclass(frozen=True)
class LikelyLitOverlay:
    key_id: int
    note_label: str
    key_color: KeyColor
    rgb: Tuple[int, int, int]
    delta: float
    saturation: float
    confidence: float


@dataclass(frozen=True)
class UnlitFrameAssessment:
    status: AssessmentStatus
    likely_lit: Tuple[LikelyLitOverlay, ...] = ()
    reason: str = ""

    @property
    def should_warn(self) -> bool:
        return self.status == "warning" and bool(self.likely_lit)


@dataclass(frozen=True)
class ExemplarCandidate:
    slot_color: KeyColor
    key_id: int
    note_label: str
    frame_index: int
    rgb: Tuple[int, int, int]
    hsv: Tuple[float, float, float]
    delta_from_unlit: float
    confidence: float
    hist: Optional[np.ndarray] = field(default=None, compare=False)


@dataclass(frozen=True)
class AssignedExemplar:
    slot: str
    rgb: Optional[Tuple[int, int, int]]
    hist: Optional[np.ndarray]
    source: Optional[ExemplarCandidate]
    enabled: bool


@dataclass(frozen=True)
class ExemplarAssignmentResult:
    assignments: Dict[str, AssignedExemplar]
    missing_slots: Tuple[str, ...]
    disabled_slots: Tuple[str, ...]
    family_count: int
    confidence: float


@dataclass(frozen=True)
class AssistedCalibrationProposal:
    baseline_frame_index: int
    unlit_assessment: UnlitFrameAssessment
    assignment_result: ExemplarAssignmentResult
    scanned_frame_count: int
    candidate_count: int
    canceled: bool = False
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExemplarScanSettings:
    coarse_stride: int = 10
    refine_radius: int = 5
    min_rgb_delta: float = 35.0
    min_saturation: float = 35.0
    max_candidates_per_key: int = 6
    early_stop_min_confidence: float = 0.5
    early_stop_min_slot_events: int = 2
    early_stop_min_slot_span_steps: int = 2
    early_stop_confirmation_steps: int = 6


@dataclass(frozen=True)
class _FamilyEarlyStopEvidence:
    hue: float
    white_events: frozenset[tuple[int, int]]
    black_events: frozenset[tuple[int, int]]


class _BoundedFrameCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max(1, max_size)
        self._frames: OrderedDict[int, Optional[np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._frames)

    def get(self, index: int, frame_provider: FrameProvider) -> Optional[np.ndarray]:
        if index in self._frames:
            self._frames.move_to_end(index)
            return self._frames[index]

        frame = frame_provider(index)
        self._frames[index] = frame
        while len(self._frames) > self._max_size:
            self._frames.popitem(last=False)
        return frame


def overlay_note_label(overlay: OverlayConfig) -> str:
    return overlay.get_full_note_name()


def overlay_key_color(overlay: OverlayConfig) -> KeyColor:
    suffix = (overlay.key_type or "")[-1:]
    return "B" if suffix == "B" else "W"


def _overlay_bounds(
    frame_rgb: np.ndarray, overlay: OverlayConfig
) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_rgb.shape[:2]
    x1 = int(overlay.x)
    y1 = int(overlay.y)
    x2 = x1 + int(overlay.width)
    y2 = y1 + int(overlay.height)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def sample_overlay_rgb(
    frame_rgb: np.ndarray, overlay: OverlayConfig
) -> Optional[Tuple[int, int, int]]:
    bounds = _overlay_bounds(frame_rgb, overlay)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    roi = frame_rgb[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    rgb = roi.mean(axis=(0, 1)).round().astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def sample_overlay_bgr(
    frame_rgb: np.ndarray, overlay: OverlayConfig
) -> Optional[np.ndarray]:
    bounds = _overlay_bounds(frame_rgb, overlay)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    roi_rgb = frame_rgb[y1:y2, x1:x2]
    if roi_rgb.size == 0:
        return None
    return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)


def capture_unlit_references_from_frame(
    frame_rgb: np.ndarray,
    overlays: Sequence[OverlayConfig],
) -> int:
    from synthesia2midi.detection.roi_utils import get_hist_feature

    calibrated = 0
    for overlay in overlays:
        rgb = sample_overlay_rgb(frame_rgb, overlay)
        bgr = sample_overlay_bgr(frame_rgb, overlay)
        if rgb is None or bgr is None:
            continue
        overlay.unlit_reference_color = rgb
        overlay.unlit_hist = get_hist_feature(bgr)
        calibrated += 1
    return calibrated


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))


def _rgb_to_hsv_tuple(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    pixel = np.array([[rgb]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def assess_unlit_frame(
    frame_rgb: np.ndarray,
    overlays: Sequence[OverlayConfig],
    *,
    min_group_delta: float = 45.0,
    min_reference_delta: float = 35.0,
    min_saturation_delta: float = 25.0,
    min_reference_saturation: float = 35.0,
    max_reported: int = 6,
) -> UnlitFrameAssessment:
    samples: list[tuple[OverlayConfig, KeyColor, Tuple[int, int, int], Tuple[float, float, float]]] = []
    for overlay in overlays:
        rgb = sample_overlay_rgb(frame_rgb, overlay)
        if rgb is None:
            continue
        samples.append((overlay, overlay_key_color(overlay), rgb, _rgb_to_hsv_tuple(rgb)))

    if len(samples) < 4:
        return UnlitFrameAssessment(status="unknown", reason="insufficient_samples")

    likely: list[LikelyLitOverlay] = []
    for key_color in ("W", "B"):
        group = [sample for sample in samples if sample[1] == key_color]
        if len(group) < 3:
            continue

        group_rgbs = np.array([sample[2] for sample in group], dtype=np.float32)
        group_sats = np.array([sample[3][1] for sample in group], dtype=np.float32)
        median_rgb = tuple(np.median(group_rgbs, axis=0).round().astype(int).tolist())
        median_sat = float(np.median(group_sats))

        for overlay, _, rgb, hsv in group:
            group_delta = _rgb_distance(rgb, median_rgb)
            reference_delta = 0.0
            if overlay.unlit_reference_color is not None:
                reference_delta = _rgb_distance(rgb, overlay.unlit_reference_color)

            saturation_delta = hsv[1] - median_sat
            strong_group_outlier = group_delta >= min_group_delta and saturation_delta >= min_saturation_delta
            strong_reference_outlier = (
                reference_delta >= min_reference_delta
                and hsv[1] >= min_reference_saturation
            )
            if not strong_group_outlier and not strong_reference_outlier:
                continue

            confidence = min(1.0, max(group_delta / 120.0, reference_delta / 120.0))
            likely.append(
                LikelyLitOverlay(
                    key_id=overlay.key_id,
                    note_label=overlay_note_label(overlay),
                    key_color=overlay_key_color(overlay),
                    rgb=rgb,
                    delta=max(group_delta, reference_delta),
                    saturation=hsv[1],
                    confidence=confidence,
                )
            )

    if not likely:
        return UnlitFrameAssessment(status="clean")

    likely.sort(key=lambda item: (-item.confidence, item.note_label, item.key_id))
    return UnlitFrameAssessment(
        status="warning",
        likely_lit=tuple(likely[:max_reported]),
        reason="color_outlier",
    )


def _candidate_from_sample(
    frame_index: int,
    overlay: OverlayConfig,
    rgb: Tuple[int, int, int],
    hist: Optional[np.ndarray],
) -> Optional[ExemplarCandidate]:
    if overlay.unlit_reference_color is None:
        return None
    delta = _rgb_distance(rgb, overlay.unlit_reference_color)
    hsv = _rgb_to_hsv_tuple(rgb)
    confidence = min(1.0, delta / 180.0)
    return ExemplarCandidate(
        slot_color=overlay_key_color(overlay),
        key_id=overlay.key_id,
        note_label=overlay_note_label(overlay),
        frame_index=frame_index,
        rgb=rgb,
        hsv=hsv,
        delta_from_unlit=delta,
        confidence=confidence,
        hist=hist,
    )


def _frame_candidate_for_overlay(
    frame_rgb: np.ndarray,
    frame_index: int,
    overlay: OverlayConfig,
    settings: ExemplarScanSettings,
) -> Optional[ExemplarCandidate]:
    rgb = sample_overlay_rgb(frame_rgb, overlay)
    bgr = sample_overlay_bgr(frame_rgb, overlay)
    if rgb is None or bgr is None or overlay.unlit_reference_color is None:
        return None
    delta = _rgb_distance(rgb, overlay.unlit_reference_color)
    hsv = _rgb_to_hsv_tuple(rgb)
    if delta < settings.min_rgb_delta or hsv[1] < settings.min_saturation:
        return None
    from synthesia2midi.detection.roi_utils import get_hist_feature

    return _candidate_from_sample(frame_index, overlay, rgb, get_hist_feature(bgr))


def _candidate_is_better(candidate: ExemplarCandidate, current: ExemplarCandidate) -> bool:
    if candidate.delta_from_unlit != current.delta_from_unlit:
        return candidate.delta_from_unlit > current.delta_from_unlit
    if candidate.confidence != current.confidence:
        return candidate.confidence > current.confidence
    return candidate.frame_index < current.frame_index


def _circular_hue_distance(a: float, b: float) -> float:
    delta = abs(a - b)
    return min(delta, 180.0 - delta)


def _circular_mean_hue(candidates: Sequence[ExemplarCandidate]) -> float:
    angles = np.array([_family_hue(candidate) for candidate in candidates]) * (2.0 * np.pi / 180.0)
    mean_angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi
    return mean_angle * (180.0 / (2.0 * np.pi))


def _family_hue(candidate: ExemplarCandidate) -> float:
    return candidate.hsv[0] if candidate.hsv[1] > 0 else _rgb_to_hsv_tuple(candidate.rgb)[0]


def _family_sort_key(bucket: list[ExemplarCandidate]) -> tuple[int, int, float]:
    mean_hue = float(np.mean([_family_hue(item) for item in bucket]))
    cool_family = 60.0 <= mean_hue <= 140.0
    first_frame = min(item.frame_index for item in bucket)
    return (0 if cool_family else 1, first_frame, mean_hue)


def _best_candidate(candidates: Iterable[ExemplarCandidate]) -> Optional[ExemplarCandidate]:
    ordered = sorted(candidates, key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    return ordered[0] if ordered else None


def _cluster_exemplar_families(
    candidates: Sequence[ExemplarCandidate],
    family_hue_threshold: float,
) -> list[list[ExemplarCandidate]]:
    family_buckets: list[list[ExemplarCandidate]] = []
    ordered_candidates = sorted(
        candidates,
        key=lambda item: (-item.confidence, item.frame_index, item.key_id),
    )
    for candidate in ordered_candidates:
        hue = _family_hue(candidate)
        target_bucket: Optional[list[ExemplarCandidate]] = None
        for bucket in family_buckets:
            bucket_hue = float(np.mean([_family_hue(item) for item in bucket]))
            if _circular_hue_distance(hue, bucket_hue) <= family_hue_threshold:
                target_bucket = bucket
                break
        if target_bucket is None:
            if len(family_buckets) >= 2:
                continue
            target_bucket = []
            family_buckets.append(target_bucket)
        target_bucket.append(candidate)

    family_buckets.sort(key=_family_sort_key)
    return family_buckets


def _assign_exemplar_slots_with_warnings(
    candidates: Sequence[ExemplarCandidate],
    *,
    family_hue_threshold: float = 22.0,
    saved_anchors: SavedFamilyAnchors | None = None,
) -> tuple[ExemplarAssignmentResult, Tuple[str, ...]]:
    evidence_sources: dict[FamilyEvidence, list[ExemplarCandidate]] = {}
    evidence: list[FamilyEvidence] = []
    for candidate in candidates:
        item = FamilyEvidence(
            frame_index=candidate.frame_index,
            key_id=candidate.key_id,
            morphology="natural" if candidate.slot_color == "W" else "accidental",
            rgb=candidate.rgb,
            score=candidate.confidence,
        )
        evidence.append(item)
        evidence_sources.setdefault(item, []).append(candidate)

    def source_sort_key(candidate: ExemplarCandidate) -> tuple[object, ...]:
        histogram = candidate.hist
        histogram_key: tuple[object, ...] = (1,)
        if histogram is not None:
            array = np.asarray(histogram)
            histogram_key = (0, array.dtype.str, array.shape, array.tobytes())
        return (
            -candidate.delta_from_unlit,
            candidate.note_label,
            candidate.hsv,
            histogram_key,
        )

    family_assignments, warnings = assign_family_slots(
        evidence,
        saved_anchors=saved_anchors,
        family_hue_threshold=family_hue_threshold,
    )
    by_family_number = {
        assignment.family_number: assignment for assignment in family_assignments
    }
    assignments: Dict[str, AssignedExemplar] = {}
    missing: list[str] = []
    disabled: list[str] = []
    confidences: list[float] = []

    for family in COLOR_FAMILIES:
        family_number = family.number
        natural_slot, accidental_slot = slots_for_family(family_number)
        family_assignment = by_family_number.get(family_number)
        family_present = family_assignment is not None
        selected_evidence = (
            (family_assignment.natural, family_assignment.accidental)
            if family_assignment is not None
            else (None, None)
        )
        for slot, item in zip((natural_slot, accidental_slot), selected_evidence):
            source_candidates = evidence_sources.get(item, []) if item is not None else []
            source = min(source_candidates, key=source_sort_key) if source_candidates else None
            if source is None:
                assignments[slot] = AssignedExemplar(
                    slot=slot,
                    rgb=None,
                    hist=None,
                    source=None,
                    enabled=family_present,
                )
                if family_present:
                    missing.append(slot)
                else:
                    disabled.append(slot)
                continue
            assignments[slot] = AssignedExemplar(
                slot=slot,
                rgb=source.rgb,
                hist=source.hist,
                source=source,
                enabled=True,
            )
            confidences.append(source.confidence)

    assert tuple(assignments) == SUPPORTED_EXEMPLAR_SLOTS

    confidence = float(np.mean(confidences)) if confidences else 0.0
    return (
        ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=tuple(missing),
            disabled_slots=tuple(disabled),
            family_count=len(family_assignments),
            confidence=confidence,
        ),
        warnings,
    )


def assign_exemplar_slots(
    candidates: Sequence[ExemplarCandidate],
    *,
    family_hue_threshold: float = 22.0,
    saved_anchors: SavedFamilyAnchors | None = None,
) -> ExemplarAssignmentResult:
    result, _warnings = _assign_exemplar_slots_with_warnings(
        candidates,
        family_hue_threshold=family_hue_threshold,
        saved_anchors=saved_anchors,
    )
    return result


def apply_assisted_calibration_proposal(
    app_state: AppState,
    proposal: AssistedCalibrationProposal,
) -> None:
    for slot, assignment in proposal.assignment_result.assignments.items():
        app_state.detection.exemplar_key_type_enabled[slot] = assignment.enabled
        app_state.detection.exemplar_lit_colors[slot] = assignment.rgb if assignment.enabled else None
        app_state.detection.exemplar_lit_histograms[slot] = (
            assignment.hist if assignment.enabled else None
        )
    app_state.unsaved_changes = True


def build_assisted_calibration_proposal(
    frame_provider: FrameProvider,
    overlays: Sequence[OverlayConfig],
    baseline_frame_index: int,
    end_frame: int,
    settings: ExemplarScanSettings = ExemplarScanSettings(),
    progress_callback: Optional[ProgressCallback] = None,
    saved_anchors: SavedFamilyAnchors | None = None,
) -> AssistedCalibrationProposal:
    baseline_frame = frame_provider(baseline_frame_index)
    assessment = (
        assess_unlit_frame(baseline_frame, overlays)
        if baseline_frame is not None
        else UnlitFrameAssessment(status="unknown", reason="baseline frame unavailable")
    )
    if assessment.status == "unknown" and assessment.reason == "insufficient_samples":
        assessment = UnlitFrameAssessment(status="clean")
    candidates, scanned, canceled = scan_lit_exemplar_candidates(
        frame_provider,
        overlays,
        baseline_frame_index + 1,
        end_frame,
        settings=settings,
        progress_callback=progress_callback,
    )
    assignment, warnings = _assign_exemplar_slots_with_warnings(
        candidates,
        saved_anchors=saved_anchors,
    )
    return AssistedCalibrationProposal(
        baseline_frame_index=baseline_frame_index,
        unlit_assessment=assessment,
        assignment_result=assignment,
        scanned_frame_count=scanned,
        candidate_count=len(candidates),
        warnings=warnings,
        canceled=canceled,
    )


def _flatten_scan_candidates(
    candidates_by_key: dict[int, list[ExemplarCandidate]],
    active_candidates_by_key: dict[int, ExemplarCandidate],
    max_candidates_per_key: int,
) -> list[ExemplarCandidate]:
    flattened = [candidate for bucket in candidates_by_key.values() for candidate in bucket]
    flattened.extend(active_candidates_by_key.values())
    flattened.sort(key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    if max_candidates_per_key <= 0:
        return flattened

    pruned: list[ExemplarCandidate] = []
    per_key_counts: dict[int, int] = {}
    for candidate in flattened:
        count = per_key_counts.get(candidate.key_id, 0)
        if count >= max_candidates_per_key:
            continue
        per_key_counts[candidate.key_id] = count + 1
        pruned.append(candidate)
    return pruned


def _store_completed_candidate(
    candidates_by_key: dict[int, list[ExemplarCandidate]],
    candidate: ExemplarCandidate,
    max_candidates_per_key: int,
) -> None:
    bucket = candidates_by_key.setdefault(candidate.key_id, [])
    bucket.append(candidate)
    if max_candidates_per_key <= 0 or len(bucket) <= max_candidates_per_key:
        return
    bucket.sort(key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    del bucket[max_candidates_per_key:]


def _store_recent_candidate(
    recent_candidates_by_key: dict[int, list[ExemplarCandidate]],
    candidate: ExemplarCandidate,
    history_size: int,
) -> None:
    bucket = recent_candidates_by_key.setdefault(candidate.key_id, [])
    bucket.append(candidate)
    if len(bucket) > history_size:
        del bucket[:-history_size]


def _complete_two_family_evidence(
    candidates: Sequence[ExemplarCandidate],
    settings: ExemplarScanSettings,
    stride: int,
) -> Optional[tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence]]:
    family_buckets = _cluster_exemplar_families(candidates, 22.0)
    if len(family_buckets) != 2:
        return None

    required_events = max(1, settings.early_stop_min_slot_events)
    required_span = stride * max(0, settings.early_stop_min_slot_span_steps)
    family_evidence: list[_FamilyEarlyStopEvidence] = []
    for family in family_buckets:
        events_by_color: dict[KeyColor, frozenset[tuple[int, int]]] = {}
        for key_color in ("W", "B"):
            supporting_candidates = [
                candidate
                for candidate in family
                if candidate.slot_color == key_color
                and candidate.confidence >= settings.early_stop_min_confidence
            ]
            supporting_frames = sorted({candidate.frame_index for candidate in supporting_candidates})
            if len(supporting_frames) < required_events:
                return None
            if supporting_frames[-1] - supporting_frames[0] < required_span:
                return None
            events_by_color[key_color] = frozenset(
                (candidate.key_id, candidate.frame_index)
                for candidate in supporting_candidates
            )
        family_evidence.append(
            _FamilyEarlyStopEvidence(
                hue=_circular_mean_hue(family),
                white_events=events_by_color["W"],
                black_events=events_by_color["B"],
            )
        )
    return family_evidence[0], family_evidence[1]


def _match_family_evidence(
    current: tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence],
    initial: tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence],
    family_hue_threshold: float = 22.0,
) -> Optional[tuple[tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence], ...]]:
    direct = ((current[0], initial[0]), (current[1], initial[1]))
    swapped = ((current[1], initial[0]), (current[0], initial[1]))
    matched = min(
        (direct, swapped),
        key=lambda pairs: sum(
            _circular_hue_distance(current_family.hue, initial_family.hue)
            for current_family, initial_family in pairs
        ),
    )
    if any(
        _circular_hue_distance(current_family.hue, initial_family.hue)
        > family_hue_threshold
        for current_family, initial_family in matched
    ):
        return None
    return matched


def scan_lit_exemplar_candidates(
    frame_provider: FrameProvider,
    overlays: Sequence[OverlayConfig],
    start_frame: int,
    end_frame: int,
    *,
    settings: ExemplarScanSettings = ExemplarScanSettings(),
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[list[ExemplarCandidate], int, bool]:
    candidates_by_key: dict[int, list[ExemplarCandidate]] = {}
    recent_candidates_by_key: dict[int, list[ExemplarCandidate]] = {}
    active_candidates_by_key: dict[int, ExemplarCandidate] = {}
    scanned = 0
    end_frame = max(start_frame, end_frame)
    stride = max(1, settings.coarse_stride)
    frame_cache = _BoundedFrameCache(stride + (2 * settings.refine_radius) + 1)
    complete_since_frame: Optional[int] = None
    confirmation_evidence: Optional[
        tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence]
    ] = None
    complete_evidence: Optional[
        tuple[_FamilyEarlyStopEvidence, _FamilyEarlyStopEvidence]
    ] = None
    recent_history_size = max(2, settings.early_stop_min_slot_events + 1)

    for frame_index in range(start_frame, end_frame + 1, stride):
        if progress_callback is not None and not progress_callback(frame_index, end_frame):
            return [], scanned, True

        frame = frame_cache.get(frame_index, frame_provider)
        if frame is None:
            continue
        scanned += 1
        recent_history_changed = False

        coarse_candidates: dict[int, ExemplarCandidate] = {}
        for overlay in overlays:
            coarse_candidate = _frame_candidate_for_overlay(frame, frame_index, overlay, settings)
            if coarse_candidate is None:
                continue
            coarse_candidates[overlay.key_id] = coarse_candidate

        for overlay in overlays:
            key_id = overlay.key_id
            best = coarse_candidates.get(key_id)
            active = active_candidates_by_key.get(key_id)

            refine_start = max(start_frame, frame_index - settings.refine_radius)
            refine_end = min(end_frame, frame_index + settings.refine_radius)
            for refined_index in range(refine_start, refine_end + 1):
                refined_frame = frame_cache.get(refined_index, frame_provider)
                if refined_frame is None:
                    continue
                refined_candidate = _frame_candidate_for_overlay(
                    refined_frame,
                    refined_index,
                    overlay,
                    settings,
                    )
                if refined_candidate is not None and (
                    best is None or _candidate_is_better(refined_candidate, best)
                ):
                    best = refined_candidate

            if best is None:
                if active is not None:
                    _store_completed_candidate(
                        candidates_by_key,
                        active,
                        settings.max_candidates_per_key,
                    )
                    _store_recent_candidate(
                        recent_candidates_by_key,
                        active,
                        recent_history_size,
                    )
                    recent_history_changed = True
                    del active_candidates_by_key[key_id]
                continue

            if active is None:
                active_candidates_by_key[key_id] = best
            elif _candidate_is_better(best, active):
                active_candidates_by_key[key_id] = best

        if recent_history_changed:
            recent_completed_candidates = _flatten_scan_candidates(
                recent_candidates_by_key,
                {},
                0,
            )
            complete_evidence = _complete_two_family_evidence(
                recent_completed_candidates,
                settings,
                stride,
            )
        if complete_evidence is not None:
            if complete_since_frame is None:
                complete_since_frame = frame_index
                confirmation_evidence = complete_evidence
            elif confirmation_evidence is not None:
                matched_evidence = _match_family_evidence(
                    complete_evidence,
                    confirmation_evidence,
                )
                if matched_evidence is None:
                    complete_since_frame = frame_index
                    confirmation_evidence = complete_evidence
                else:
                    confirmation_span = stride * max(
                        0,
                        settings.early_stop_confirmation_steps,
                    )
                    has_fresh_slot_evidence = all(
                        current.white_events - initial.white_events
                        and current.black_events - initial.black_events
                        for current, initial in matched_evidence
                    )
                    if (
                        frame_index - complete_since_frame >= confirmation_span
                        and has_fresh_slot_evidence
                    ):
                        return (
                            _flatten_scan_candidates(
                                candidates_by_key,
                                active_candidates_by_key,
                                settings.max_candidates_per_key,
                            ),
                            scanned,
                            False,
                        )
        else:
            complete_since_frame = None
            confirmation_evidence = None

    return (
        _flatten_scan_candidates(
            candidates_by_key,
            active_candidates_by_key,
            settings.max_candidates_per_key,
        ),
        scanned,
        False,
    )
