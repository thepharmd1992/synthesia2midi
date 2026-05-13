import cv2
import numpy as np

from synthesia2midi.detection.monolithic_detector import MonolithicPianoDetector


LEGACY_DETECTOR_METHODS = {
    "detect_keys",
    "assign_notes",
    "create_final_visualization",
    "run_complete_detection",
    "_add_overlay_padding",
    "_detect_black_keys",
    "_maybe_recover_black_keys",
    "_threshold_black_region",
    "_find_white_strip_start",
    "_runs_from_mask",
    "_estimate_white_key_width",
    "_find_white_valley_centers",
    "_guided_split_white_span",
    "_classify_large_center_gaps",
    "_split_span_evenly",
    "_score_black_note_center_map",
    "_build_black_note_center_map",
    "_estimate_black_residual_samples",
    "_apply_black_residual_edge_warp",
    "_estimate_white_centers_from_d_lattice",
    "_build_white_spans_from_centers",
    "_detect_white_keys_from_black_d_lattice",
    "_detect_white_keys_from_black_boundary_solver",
    "_detect_white_keys_from_black",
    "_detect_white_keys",
    "_trim_white_key_top",
    "_extract_note_name",
    "_extract_note_octave",
    "_apply_white_post_assignment_adjustments",
    "_find_confident_f_sharp_anchor",
    "_find_f_sharp_anchor_candidates",
    "_assign_notes_type_aware",
    "_assign_notes_chromatically_from_anchor",
    "_assign_black_key_notes",
    "_assign_white_key_notes_by_scanning",
    "_fallback_note_assignment",
    "_fallback_white_assignment",
}


def _write_image(tmp_path, image):
    path = tmp_path / "keyboard.jpg"
    assert cv2.imwrite(str(path), image)
    return path


def test_legacy_detector_method_surface_is_preserved():
    for method_name in LEGACY_DETECTOR_METHODS:
        assert callable(getattr(MonolithicPianoDetector, method_name))


def test_detect_keys_characterizes_synthetic_keyboard(tmp_path):
    image = np.full((120, 320, 3), 245, dtype=np.uint8)
    for x in range(0, 321, 20):
        cv2.line(image, (x, 60), (x, 119), (120, 120, 120), 2)
    for x in [10, 25, 60, 75, 90, 125, 140, 175, 190, 205]:
        cv2.rectangle(image, (x, 0), (x + 8, 50), (0, 0, 0), -1)

    detector = MonolithicPianoDetector(
        str(_write_image(tmp_path, image)),
        keyboard_region=(0, 120, 0, 320),
        detection_profile={
            "black_min_width": 4,
            "black_max_width": 20,
            "black_column_ratio": 0.2,
            "white_sep_min_width": 1,
        },
    )

    assert detector.detect_keys() == (10, 20)
    assert detector.black_keys[:3] == [(12, 0, 5, 50), (27, 0, 5, 50), (62, 0, 5, 50)]
    assert detector.white_keys[:3] == [(3, 96, 7, 24), (17, 96, 9, 24), (34, 96, 8, 24)]


def test_type_aware_note_assignment_characterizes_anchor_scanning(tmp_path):
    image = np.full((120, 320, 3), 255, dtype=np.uint8)
    detector = MonolithicPianoDetector(
        str(_write_image(tmp_path, image)),
        keyboard_region=(0, 120, 0, 320),
    )
    detector.black_keys = [
        (10, 0, 8, 45),
        (25, 0, 8, 45),
        (60, 0, 8, 45),
        (75, 0, 8, 45),
        (90, 0, 8, 45),
        (125, 0, 8, 45),
        (140, 0, 8, 45),
        (175, 0, 8, 45),
        (190, 0, 8, 45),
        (205, 0, 8, 45),
    ]
    detector.white_keys = [
        (0, 50, 16, 60),
        (20, 50, 16, 60),
        (40, 50, 16, 60),
        (55, 50, 16, 60),
        (75, 50, 16, 60),
        (95, 50, 16, 60),
        (115, 50, 16, 60),
        (135, 50, 16, 60),
        (155, 50, 16, 60),
        (170, 50, 16, 60),
        (190, 50, 16, 60),
        (210, 50, 16, 60),
        (230, 50, 16, 60),
        (250, 50, 16, 60),
    ]

    assert detector._find_f_sharp_anchor_candidates() == [2]

    notes = detector.assign_notes()

    assert notes[64]["note"] == "F#0"
    assert notes[64]["type"] == "black"
    assert notes[179]["note"] == "F#1"
    assert notes[8]["note"] == "C0"
    assert notes[63]["note"] == "F0"
    assert len(notes) == 24
