"""Deterministic clustering and slot assignment for lit color families."""

from __future__ import annotations

import colorsys
import heapq
import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


Morphology = Literal["natural", "accidental"]
RGB = tuple[int, int, int]
SavedFamilyAnchors = Mapping[int, Mapping[Morphology, RGB | None]]

# HSV hue values use OpenCV's 0-180 scale, so 11 units represent 22 degrees.
DEFAULT_FAMILY_HUE_THRESHOLD = 11.0
DEFAULT_MIN_TEMPORAL_SEPARATION = 2
DEFAULT_MIN_SATURATION = 35.0
DEFAULT_MIN_VALUE = 20.0
MAX_COLOR_FAMILIES = 4

TOO_MANY_FAMILIES_WARNING = "too_many_families"
ANCHOR_CONFLICT_WARNING = "anchor_conflict"


@dataclass(frozen=True)
class FamilyEvidence:
    frame_index: int
    key_id: int
    morphology: Morphology
    rgb: RGB
    score: float


@dataclass
class FamilyAssignment:
    family_number: int
    natural: FamilyEvidence | None = None
    accidental: FamilyEvidence | None = None
    confidence: float = 0.0

    @property
    def complete(self) -> bool:
        return self.natural is not None and self.accidental is not None


def _rgb_to_hsv(rgb: RGB) -> tuple[float, float, float]:
    red, green, blue = (max(0, min(255, channel)) / 255.0 for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    return hue * 180.0, saturation * 255.0, value * 255.0


def _circular_hue_distance(a: float, b: float) -> float:
    delta = abs(a - b)
    return min(delta, 180.0 - delta)


def _cluster_hue(cluster: Sequence[FamilyEvidence]) -> float:
    angles = [
        _rgb_to_hsv(item.rgb)[0] * (2.0 * math.pi / 180.0)
        for item in cluster
    ]
    weights = [max(0.01, item.score) for item in cluster]
    x = sum(math.cos(angle) * weight for angle, weight in zip(angles, weights))
    y = sum(math.sin(angle) * weight for angle, weight in zip(angles, weights))
    mean = math.atan2(y, x)
    if mean < 0:
        mean += 2.0 * math.pi
    return mean * (180.0 / (2.0 * math.pi))


def _evidence_sort_key(item: FamilyEvidence) -> tuple[object, ...]:
    return (
        item.frame_index,
        item.key_id,
        item.morphology,
        item.rgb,
        -item.score,
    )


def _cluster_sort_key(cluster: Sequence[FamilyEvidence]) -> tuple[object, ...]:
    first = min(cluster, key=_evidence_sort_key)
    return first.frame_index, _cluster_hue(cluster), first.rgb


def _passes_color_guards(
    evidence: FamilyEvidence,
    min_saturation: float,
    min_value: float,
) -> bool:
    _hue, saturation, value = _rgb_to_hsv(evidence.rgb)
    return saturation >= min_saturation and value >= min_value


def _is_temporally_stable(
    cluster: Sequence[FamilyEvidence],
    min_temporal_separation: int,
) -> bool:
    frames = sorted({item.frame_index for item in cluster})
    return any(
        right - left > max(0, min_temporal_separation)
        for left, right in zip(frames, frames[1:])
    )


def cluster_family_evidence(
    evidence: Sequence[FamilyEvidence],
    *,
    family_hue_threshold: float = DEFAULT_FAMILY_HUE_THRESHOLD,
    min_temporal_separation: int = DEFAULT_MIN_TEMPORAL_SEPARATION,
    min_saturation: float = DEFAULT_MIN_SATURATION,
    min_value: float = DEFAULT_MIN_VALUE,
) -> list[list[FamilyEvidence]]:
    """Agglomerate guarded evidence and return only stable color families."""

    clusters = {
        cluster_id: [item]
        for cluster_id, item in enumerate(
            item
            for item in sorted(evidence, key=_evidence_sort_key)
            if _passes_color_guards(item, min_saturation, min_value)
        )
    }
    versions = {cluster_id: 0 for cluster_id in clusters}
    merge_candidates: list[tuple[object, ...]] = []

    def add_merge_candidate(left_id: int, right_id: int) -> None:
        if left_id > right_id:
            left_id, right_id = right_id, left_id
        left = clusters[left_id]
        right = clusters[right_id]
        distance = _circular_hue_distance(_cluster_hue(left), _cluster_hue(right))
        if distance > family_hue_threshold:
            return
        heapq.heappush(
            merge_candidates,
            (
                distance,
                _cluster_sort_key(left),
                _cluster_sort_key(right),
                left_id,
                right_id,
                versions[left_id],
                versions[right_id],
            ),
        )

    cluster_ids = list(clusters)
    for left_index, left_id in enumerate(cluster_ids):
        for right_id in cluster_ids[left_index + 1 :]:
            add_merge_candidate(left_id, right_id)

    while merge_candidates:
        (
            _distance,
            _left_key,
            _right_key,
            left_id,
            right_id,
            left_version,
            right_version,
        ) = heapq.heappop(merge_candidates)
        if (
            left_id not in clusters
            or right_id not in clusters
            or versions[left_id] != left_version
            or versions[right_id] != right_version
        ):
            continue

        clusters[left_id] = sorted(
            [*clusters[left_id], *clusters[right_id]],
            key=_evidence_sort_key,
        )
        versions[left_id] += 1
        del clusters[right_id]
        del versions[right_id]

        for other_id in clusters:
            if other_id != left_id:
                add_merge_candidate(left_id, other_id)

    stable = [
        cluster
        for cluster in clusters.values()
        if _is_temporally_stable(cluster, min_temporal_separation)
    ]
    return sorted(stable, key=_cluster_sort_key)


def _best_evidence(
    cluster: Sequence[FamilyEvidence], morphology: Morphology
) -> FamilyEvidence | None:
    compatible = [item for item in cluster if item.morphology == morphology]
    if not compatible:
        return None
    return min(
        compatible,
        key=lambda item: (-item.score, item.frame_index, item.key_id, item.rgb),
    )


def _cluster_confidence(cluster: Sequence[FamilyEvidence]) -> float:
    average_score = sum(max(0.0, min(1.0, item.score)) for item in cluster) / len(
        cluster
    )
    distinct_key_bonus = min(0.1, 0.05 * (len({item.key_id for item in cluster}) - 1))
    return min(1.0, average_score + distinct_key_bonus)


def _cluster_strength_key(cluster: Sequence[FamilyEvidence]) -> tuple[object, ...]:
    return (
        -_cluster_confidence(cluster),
        -len({item.frame_index for item in cluster}),
        -len({item.key_id for item in cluster}),
        _cluster_sort_key(cluster),
    )


def _anchor_distance(rgb: RGB, anchor: RGB) -> float:
    return _circular_hue_distance(_rgb_to_hsv(rgb)[0], _rgb_to_hsv(anchor)[0])


def _matching_anchor_number(
    cluster: Sequence[FamilyEvidence],
    saved_anchors: SavedFamilyAnchors,
    family_hue_threshold: float,
) -> tuple[int | None, bool]:
    nearest_by_morphology: list[int] = []
    for morphology in ("natural", "accidental"):
        sample = _best_evidence(cluster, morphology)
        if sample is None:
            continue
        compatible = [
            (_anchor_distance(sample.rgb, anchor), family_number)
            for family_number, anchors in saved_anchors.items()
            if (anchor := anchors.get(morphology)) is not None
            and 1 <= family_number <= MAX_COLOR_FAMILIES
        ]
        if not compatible:
            continue
        distance, family_number = min(compatible, key=lambda item: (item[0], item[1]))
        if distance <= family_hue_threshold:
            nearest_by_morphology.append(family_number)

    matched_numbers = set(nearest_by_morphology)
    if len(matched_numbers) > 1:
        return None, True
    if not matched_numbers:
        return None, False
    return matched_numbers.pop(), False


def assign_family_slots(
    evidence: Sequence[FamilyEvidence],
    *,
    saved_anchors: SavedFamilyAnchors | None = None,
    family_hue_threshold: float = DEFAULT_FAMILY_HUE_THRESHOLD,
    min_temporal_separation: int = DEFAULT_MIN_TEMPORAL_SEPARATION,
) -> tuple[list[FamilyAssignment], tuple[str, ...]]:
    """Assign up to four stable clusters to deterministic family identities."""

    clusters = cluster_family_evidence(
        evidence,
        family_hue_threshold=family_hue_threshold,
        min_temporal_separation=min_temporal_separation,
    )
    warnings: list[str] = []
    if len(clusters) > MAX_COLOR_FAMILIES:
        warnings.append(TOO_MANY_FAMILIES_WARNING)
        clusters = sorted(clusters, key=_cluster_strength_key)[:MAX_COLOR_FAMILIES]

    anchors = saved_anchors or {}
    reserved_numbers = {
        number
        for number, morphology_anchors in anchors.items()
        if 1 <= number <= MAX_COLOR_FAMILIES
        and any(anchor is not None for anchor in morphology_anchors.values())
    }
    anchored: dict[int, Sequence[FamilyEvidence]] = {}
    unmatched: list[Sequence[FamilyEvidence]] = []

    for cluster in sorted(clusters, key=_cluster_strength_key):
        family_number, conflict = _matching_anchor_number(
            cluster,
            anchors,
            family_hue_threshold,
        )
        if conflict or family_number in anchored:
            if ANCHOR_CONFLICT_WARNING not in warnings:
                warnings.append(ANCHOR_CONFLICT_WARNING)
            unmatched.append(cluster)
        elif family_number is None:
            unmatched.append(cluster)
        else:
            anchored[family_number] = cluster

    available_numbers = [
        number
        for number in range(1, MAX_COLOR_FAMILIES + 1)
        if number not in reserved_numbers and number not in anchored
    ]
    for cluster in sorted(unmatched, key=_cluster_sort_key):
        if not available_numbers:
            break
        anchored[available_numbers.pop(0)] = cluster

    assignments = [
        FamilyAssignment(
            family_number=family_number,
            natural=_best_evidence(cluster, "natural"),
            accidental=_best_evidence(cluster, "accidental"),
            confidence=_cluster_confidence(cluster),
        )
        for family_number, cluster in sorted(anchored.items())
    ]
    return assignments, tuple(warnings)
