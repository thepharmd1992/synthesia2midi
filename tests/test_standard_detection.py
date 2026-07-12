import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.detection.roi_utils import get_hist_feature
from synthesia2midi.detection.standard import StandardDetection
from synthesia2midi.workflows.conversion import _midi_channel_for_exemplar


def _overlay(
    key_id: int = 1,
    *,
    key_type: str = "LW",
    x: int = 0,
    unlit_hist: np.ndarray | None = None,
) -> OverlayConfig:
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave="C",
        x=x,
        y=0,
        width=4,
        height=4,
        key_type=key_type,
        unlit_reference_color=(0, 0, 0),
        unlit_hist=unlit_hist,
    )


def _solid_rgb(rgb: tuple[int, int, int], *, width: int = 4) -> np.ndarray:
    return np.full((4, width, 3), rgb[::-1], dtype=np.uint8)


def _color_exemplars(**updates):
    exemplars = {
        "LW": None,
        "LB": None,
        "RW": None,
        "RB": None,
        "COLOR_3_W": None,
        "COLOR_3_B": None,
        "COLOR_4_W": None,
        "COLOR_4_B": None,
    }
    exemplars.update(updates)
    return exemplars


def _histogram_exemplars(**updates):
    exemplars = {slot: None for slot in _color_exemplars()}
    exemplars.update(updates)
    return exemplars


def test_strongest_natural_exemplar_slot_is_retained_for_pressed_key():
    detector = StandardDetection()
    overlay = _overlay()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((120, 30, 160)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            LW=(255, 255, 255),
            RW=(255, 0, 0),
            COLOR_3_W=(220, 180, 0),
            COLOR_4_W=(120, 30, 160),
        ),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    assert pressed == {overlay.key_id}
    assert detector.get_last_exemplar_match(overlay.key_id) == "COLOR_4_W"


def test_exact_yellow_uses_closest_color_three_identity_and_channel():
    detector = StandardDetection()
    overlay = _overlay()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((255, 255, 0)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            LW=(255, 128, 0),
            COLOR_3_W=(255, 255, 0),
        ),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    winning_slot = detector.get_last_exemplar_match(overlay.key_id)
    assert pressed == {overlay.key_id}
    assert overlay.last_progression_ratio > 1.2
    assert winning_slot == "COLOR_3_W"
    assert _midi_channel_for_exemplar(winning_slot) == 2


def test_legacy_hand_hue_classification_does_not_hide_color_two_family():
    detector = StandardDetection()
    overlay = _overlay()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((0, 0, 255)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            LW=(255, 0, 0),
            RW=(0, 0, 255),
        ),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
        hand_assignment_enabled=True,
        hand_detection_calibrated=True,
        left_hand_hue_mean=120.0,
        right_hand_hue_mean=0.0,
    )

    assert pressed == {overlay.key_id}
    assert detector.get_last_exemplar_match(overlay.key_id) == "RW"


def test_identity_uses_only_color_exemplars_that_pass_detection():
    detector = StandardDetection()
    overlay = _overlay()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((100, 0, 0)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            LW=(130, 0, 0),
            RW=(80, 80, 0),
        ),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    assert pressed == {overlay.key_id}
    assert detector.get_last_exemplar_match(overlay.key_id) == "RW"


def test_accidental_overlay_never_matches_natural_slot_when_hand_hues_are_close():
    detector = StandardDetection()
    overlay = _overlay(key_type="LB")

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((100, 0, 0)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(LW=(100, 0, 0)),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
        hand_assignment_enabled=True,
        hand_detection_calibrated=True,
        left_hand_hue_mean=10.0,
        right_hand_hue_mean=12.0,
    )

    assert pressed == set()
    assert detector.get_last_exemplar_match(overlay.key_id) is None


def test_filtered_and_nonpressed_keys_never_match_an_exemplar_slot():
    detector = StandardDetection()
    retained = _overlay(key_id=1, key_type="LB", x=0)
    filtered = _overlay(key_id=3, key_type="LB", x=4)
    retained.last_progression_ratio = 0.8
    retained.last_is_lit = True
    filtered.last_progression_ratio = 0.2

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((100, 0, 0), width=8),
        overlays=[retained, filtered],
        exemplar_lit_colors=_color_exemplars(COLOR_3_B=(100, 0, 0)),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=True,
    )

    assert pressed == {retained.key_id}
    assert detector.get_last_exemplar_match(retained.key_id) == "COLOR_3_B"
    assert detector.get_last_exemplar_match(filtered.key_id) is None

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((0, 0, 0)),
        overlays=[retained],
        exemplar_lit_colors=_color_exemplars(COLOR_3_B=(100, 0, 0)),
        exemplar_lit_histograms=_histogram_exemplars(),
        detection_threshold=0.8,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    assert pressed == set()
    assert detector.get_last_exemplar_match(retained.key_id) is None


def test_histogram_fallback_retains_highest_valid_winning_slot():
    unlit_roi = _solid_rgb((0, 0, 0))
    mixed_roi = unlit_roi.copy()
    mixed_roi[0, :, :] = (255, 255, 255)
    overlay = _overlay(unlit_hist=get_hist_feature(unlit_roi))
    detector = StandardDetection()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((40, 0, 40)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            LW=(255, 255, 255),
            RW=(255, 255, 255),
            COLOR_3_W=(255, 255, 255),
            COLOR_4_W=(255, 255, 255),
        ),
        exemplar_lit_histograms=_histogram_exemplars(
            LW=get_hist_feature(_solid_rgb((0, 0, 255))),
            RW=get_hist_feature(_solid_rgb((0, 255, 0))),
            COLOR_3_W=get_hist_feature(mixed_roi),
            COLOR_4_W=get_hist_feature(_solid_rgb((255, 0, 0))),
        ),
        detection_threshold=0.4,
        hist_ratio_threshold=0.8,
        use_histogram_detection=True,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    assert pressed == {overlay.key_id}
    assert detector.get_last_exemplar_match(overlay.key_id) == "COLOR_3_W"


def test_color_winner_takes_precedence_when_histogram_also_passes():
    unlit_roi = _solid_rgb((0, 0, 0))
    mixed_roi = unlit_roi.copy()
    mixed_roi[0, :, :] = (255, 255, 255)
    overlay = _overlay(unlit_hist=get_hist_feature(unlit_roi))
    detector = StandardDetection()

    pressed = detector.detect_frame(
        frame_bgr=_solid_rgb((160, 0, 0)),
        overlays=[overlay],
        exemplar_lit_colors=_color_exemplars(
            COLOR_3_W=(255, 255, 255),
            COLOR_4_W=(200, 0, 0),
        ),
        exemplar_lit_histograms=_histogram_exemplars(
            COLOR_3_W=get_hist_feature(mixed_roi),
            COLOR_4_W=get_hist_feature(_solid_rgb((255, 0, 0))),
        ),
        detection_threshold=0.7,
        hist_ratio_threshold=0.8,
        use_histogram_detection=True,
        use_delta_detection=False,
        apply_black_filter=False,
    )

    assert pressed == {overlay.key_id}
    assert detector.get_last_exemplar_match(overlay.key_id) == "COLOR_4_W"
