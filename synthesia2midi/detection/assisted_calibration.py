"""Pure data models and overlay sampling helpers for assisted calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig

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


@dataclass(frozen=True)
class ExemplarScanSettings:
    coarse_stride: int = 10
    refine_radius: int = 5
    min_rgb_delta: float = 35.0
    min_saturation: float = 35.0
    max_candidates_per_key: int = 6


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
    scanned = 0
    end_frame = max(start_frame, end_frame)
    stride = max(1, settings.coarse_stride)

    for frame_index in range(start_frame, end_frame + 1, stride):
        if progress_callback is not None and not progress_callback(frame_index, end_frame):
            return [], scanned, True

        frame = frame_provider(frame_index)
        if frame is None:
            continue
        scanned += 1

        coarse_candidates: dict[int, ExemplarCandidate] = {}
        for overlay in overlays:
            coarse_candidate = _frame_candidate_for_overlay(frame, frame_index, overlay, settings)
            if coarse_candidate is None:
                continue
            coarse_candidates[overlay.key_id] = coarse_candidate

        if not coarse_candidates:
            continue

        for overlay in overlays:
            best = coarse_candidates.get(overlay.key_id)
            refine_start = max(start_frame, frame_index - settings.refine_radius)
            refine_end = min(end_frame, frame_index + settings.refine_radius)
            for refined_index in range(refine_start, refine_end + 1):
                refined_frame = frame_provider(refined_index)
                if refined_frame is None:
                    continue
                refined_candidate = _frame_candidate_for_overlay(
                    refined_frame,
                    refined_index,
                    overlay,
                    settings,
                )
                if refined_candidate is not None and (
                    best is None or refined_candidate.confidence > best.confidence
                ):
                    best = refined_candidate

            if best is None:
                continue

            bucket = candidates_by_key.setdefault(overlay.key_id, [])
            bucket.append(best)
            bucket.sort(key=lambda item: (-item.confidence, item.frame_index))
            del bucket[settings.max_candidates_per_key :]

    flattened = [candidate for bucket in candidates_by_key.values() for candidate in bucket]
    flattened.sort(key=lambda item: (-item.confidence, item.frame_index, item.key_id))
    return flattened, scanned, False
