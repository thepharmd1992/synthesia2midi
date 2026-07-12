import pytest

import synthesia2midi.detection.color_family_assignment as assignment_module
from synthesia2midi.detection.color_family_assignment import (
    FamilyAssignment,
    FamilyEvidence,
    assign_family_slots,
    cluster_family_evidence,
)


FAMILY_COLORS = (
    ((70, 130, 230), (45, 95, 185)),
    ((235, 65, 65), (185, 35, 35)),
    ((235, 215, 45), (185, 165, 25)),
    ((45, 210, 70), (25, 160, 50)),
    ((165, 80, 220), (115, 45, 170)),
)


def _family_evidence(
    family_index: int,
    *,
    first_frame: int,
    score: float = 0.9,
    reverse_morphologies: bool = False,
) -> list[FamilyEvidence]:
    natural_rgb, accidental_rgb = FAMILY_COLORS[family_index]
    evidence = [
        FamilyEvidence(first_frame, 10 + family_index, "natural", natural_rgb, score),
        FamilyEvidence(first_frame + 1, 20 + family_index, "accidental", accidental_rgb, score),
        FamilyEvidence(first_frame + 10, 30 + family_index, "natural", natural_rgb, score),
        FamilyEvidence(first_frame + 11, 40 + family_index, "accidental", accidental_rgb, score),
    ]
    return list(reversed(evidence)) if reverse_morphologies else evidence


@pytest.mark.parametrize("family_count", [1, 2, 3, 4])
def test_assigns_one_through_four_stable_families(family_count):
    evidence: list[FamilyEvidence] = []
    for family_index in range(family_count):
        evidence.extend(_family_evidence(family_index, first_frame=family_index * 20))

    assignments, warnings = assign_family_slots(list(reversed(evidence)))

    assert [assignment.family_number for assignment in assignments] == list(
        range(1, family_count + 1)
    )
    assert all(isinstance(assignment, FamilyAssignment) for assignment in assignments)
    assert all(assignment.complete for assignment in assignments)
    assert [assignment.natural.rgb for assignment in assignments] == [
        colors[0] for colors in FAMILY_COLORS[:family_count]
    ]
    assert [assignment.accidental.rgb for assignment in assignments] == [
        colors[1] for colors in FAMILY_COLORS[:family_count]
    ]
    assert warnings == ()


def test_saved_anchors_preserve_color_one_and_two_when_evidence_order_reverses():
    evidence = [
        *_family_evidence(1, first_frame=10, reverse_morphologies=True),
        *_family_evidence(0, first_frame=40, reverse_morphologies=True),
    ]
    saved_anchors = {
        1: {"natural": FAMILY_COLORS[0][0], "accidental": FAMILY_COLORS[0][1]},
        2: {"natural": FAMILY_COLORS[1][0], "accidental": FAMILY_COLORS[1][1]},
    }

    assignments, warnings = assign_family_slots(evidence, saved_anchors=saved_anchors)

    by_number = {assignment.family_number: assignment for assignment in assignments}
    assert by_number[1].natural.rgb == FAMILY_COLORS[0][0]
    assert by_number[2].natural.rgb == FAMILY_COLORS[1][0]
    assert warnings == ()


def test_single_flash_is_not_a_stable_family():
    evidence = [
        FamilyEvidence(10, 1, "natural", FAMILY_COLORS[0][0], 1.0),
        FamilyEvidence(10, 2, "accidental", FAMILY_COLORS[0][1], 1.0),
    ]

    assert cluster_family_evidence(evidence) == []
    assert assign_family_slots(evidence) == ([], ())


def test_single_multi_frame_flash_is_not_a_stable_family():
    evidence = [
        FamilyEvidence(10, 1, "natural", FAMILY_COLORS[0][0], 1.0),
        FamilyEvidence(11, 1, "natural", FAMILY_COLORS[0][0], 1.0),
        FamilyEvidence(12, 1, "natural", FAMILY_COLORS[0][0], 1.0),
    ]

    assert cluster_family_evidence(evidence) == []


def test_nearby_hues_merge_into_one_stable_family():
    evidence = [
        FamilyEvidence(10, 1, "natural", (230, 70, 75), 0.8),
        FamilyEvidence(20, 2, "natural", (225, 85, 65), 0.9),
        FamilyEvidence(30, 3, "accidental", (180, 45, 55), 0.85),
    ]

    clusters = cluster_family_evidence(evidence)
    assignments, warnings = assign_family_slots(evidence)

    assert len(clusters) == 1
    assert len(assignments) == 1
    assert assignments[0].complete is True
    assert warnings == ()


def test_five_stable_families_keep_four_strongest_and_warn():
    evidence: list[FamilyEvidence] = []
    scores = (0.95, 0.9, 0.85, 0.8, 0.25)
    for family_index, score in enumerate(scores):
        evidence.extend(
            _family_evidence(
                family_index,
                first_frame=family_index * 20,
                score=score,
            )
        )

    assignments, warnings = assign_family_slots(evidence)

    assert len(assignments) == 4
    assert {assignment.natural.rgb for assignment in assignments} == {
        colors[0] for colors in FAMILY_COLORS[:4]
    }
    assert warnings == ("More than four stable color families were found.",)


def test_low_saturation_and_dark_evidence_do_not_form_families():
    evidence = [
        FamilyEvidence(10, 1, "natural", (120, 121, 120), 1.0),
        FamilyEvidence(20, 2, "natural", (125, 124, 125), 1.0),
        FamilyEvidence(30, 3, "accidental", (8, 1, 1), 1.0),
        FamilyEvidence(40, 4, "accidental", (9, 1, 1), 1.0),
    ]

    assert cluster_family_evidence(evidence) == []


def test_distinct_keys_raise_confidence_but_are_not_required_for_stability():
    repeated_key = [
        FamilyEvidence(10, 1, "natural", FAMILY_COLORS[0][0], 0.8),
        FamilyEvidence(20, 1, "natural", FAMILY_COLORS[0][0], 0.8),
    ]
    distinct_keys = [
        FamilyEvidence(10, 1, "natural", FAMILY_COLORS[1][0], 0.8),
        FamilyEvidence(20, 2, "natural", FAMILY_COLORS[1][0], 0.8),
    ]

    repeated_assignments, _ = assign_family_slots(repeated_key)
    distinct_assignments, _ = assign_family_slots(distinct_keys)

    assert len(repeated_assignments) == 1
    assert distinct_assignments[0].confidence > repeated_assignments[0].confidence


def test_conflicting_morphology_anchors_warn_without_reusing_either_identity():
    evidence = _family_evidence(0, first_frame=10)
    saved_anchors = {
        1: {"natural": FAMILY_COLORS[0][0]},
        2: {"accidental": FAMILY_COLORS[0][1]},
    }

    assignments, warnings = assign_family_slots(evidence, saved_anchors=saved_anchors)

    assert assignments[0].family_number == 3
    assert warnings == ("Evidence conflicts with two saved color family identities.",)


def test_agglomerative_clustering_keeps_distance_work_bounded(monkeypatch):
    evidence = [
        FamilyEvidence(
            frame_index=index,
            key_id=index % 88,
            morphology="natural" if index % 2 else "accidental",
            rgb=FAMILY_COLORS[index % 4][index % 2],
            score=0.9,
        )
        for index in range(200)
    ]
    distance_calls = 0
    original_distance = assignment_module._circular_hue_distance

    def count_distance_calls(left, right):
        nonlocal distance_calls
        distance_calls += 1
        return original_distance(left, right)

    monkeypatch.setattr(
        assignment_module,
        "_circular_hue_distance",
        count_distance_calls,
    )

    assert len(cluster_family_evidence(evidence)) == 4
    assert distance_calls < 50_000
