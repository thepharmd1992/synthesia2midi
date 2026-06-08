from dataclasses import fields

import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.workflows.manual_keyboard_fit import ManualFitParams, ManualKeyboardFitSession


def _overlay(key_id, note, x, y=20, width=10, height=30):
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave=note,
        x=x,
        y=y,
        width=width,
        height=height,
        key_type="LW" if note in {"A", "B", "C", "D", "E", "F", "G"} else "LB",
    )


def _state_with_overlays():
    app_state = AppState()
    app_state.overlays = [
        _overlay(1, "C", 0, width=10, height=40),
        _overlay(2, "C♯", 12, y=10, width=6, height=20),
        _overlay(3, "D", 24, width=10, height=40),
    ]
    return app_state


def _bounds(overlays):
    left = min(overlay.x for overlay in overlays)
    right = max(overlay.x + overlay.width for overlay in overlays)
    return left, right, right - left


def test_manual_fit_group_translate_moves_every_overlay_and_preserves_note_identity():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.translate_group(7, -3)

    assert [(o.key_id, o.note_name_in_octave, o.note_octave) for o in app_state.overlays] == [
        (1, "C", 4),
        (2, "C♯", 4),
        (3, "D", 4),
    ]
    assert [(o.x, o.y) for o in app_state.overlays] == pytest.approx([
        (8, 23),
        (19.6, 10),
        (32, 23),
    ])
    assert app_state.unsaved_changes is True


def test_manual_fit_keyboard_width_scales_centers_and_safe_widths_from_same_span():
    app_state = _state_with_overlays()
    baseline_span = _bounds(app_state.overlays)[2]
    session = ManualKeyboardFitSession(app_state)

    session.set_param("keyboard_width_delta", baseline_span)

    assert _bounds(app_state.overlays)[2] == pytest.approx(64)
    assert app_state.overlays[0].width == pytest.approx(16)
    assert app_state.overlays[1].width == pytest.approx(9.6)
    assert app_state.overlays[2].width == pytest.approx(16)


def test_manual_fit_edge_drift_moves_edges_more_than_center():
    app_state = AppState()
    app_state.overlays = [
        _overlay(1, "C", 0),
        _overlay(2, "D", 45),
        _overlay(3, "E", 90),
    ]
    baseline_centers = [overlay.x + overlay.width / 2 for overlay in app_state.overlays]
    session = ManualKeyboardFitSession(app_state)

    session.set_param("left_edge_drift", 20)

    shifts = [
        (overlay.x + overlay.width / 2) - baseline_center
        for overlay, baseline_center in zip(app_state.overlays, baseline_centers)
    ]
    assert shifts[0] > shifts[1] > shifts[2]


def test_manual_fit_drawn_regions_apply_visible_safe_margins():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.set_detection_region("white", 50, 100)
    session.set_detection_region("black", 10, 40)

    white_left, black, white_right = app_state.overlays
    assert (white_left.x, white_left.y, white_left.width, white_left.height) == pytest.approx((1, 57.5, 8, 35))
    assert (white_right.x, white_right.y, white_right.width, white_right.height) == pytest.approx((25, 57.5, 8, 35))
    assert (black.x, black.y, black.width, black.height) == pytest.approx((12.6, 14.5, 4.8, 21))


def test_manual_fit_keyboard_top_moves_safe_top_only():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.set_detection_region("white", 50, 100)
    session.set_detection_region("black", 10, 40)
    session.set_param("keyboard_top_delta", 10)

    white_left, black, _white_right = app_state.overlays
    assert (white_left.y, white_left.height) == pytest.approx((66, 28))
    assert (black.y, black.height) == pytest.approx((23, 14))


def test_manual_fit_removed_geometry_controls_are_not_backend_parameters():
    removed_names = {
        "white_y_delta",
        "white_width_delta",
        "black_x_delta",
        "white_height_delta",
        "black_y_delta",
        "black_height_delta",
        "white_band_top_delta",
        "white_band_bottom_delta",
        "black_band_top_delta",
        "black_band_bottom_delta",
        "white_x_inset",
        "black_x_inset",
    }

    assert removed_names.isdisjoint({field.name for field in fields(ManualFitParams)})


def test_manual_fit_single_overlay_override_survives_later_group_changes():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.move_single_overlay_by_index(0, 100, 50)
    session.translate_group(10, 0)

    assert app_state.overlays[0].x == pytest.approx(110)
    assert app_state.overlays[0].y == pytest.approx(50)
    assert app_state.overlays[1].x == pytest.approx(22.6)
    assert session.overridden_key_ids() == {1}


def test_manual_fit_control_updates_preserve_current_group_position():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.translate_group(30, 8)
    session.update_control_params(ManualFitParams(keyboard_width_delta=20))

    assert session.params.group_dx == pytest.approx(30)
    assert session.params.group_dy == pytest.approx(8)
    assert app_state.overlays[0].x > 0
    assert app_state.overlays[0].y == pytest.approx(34)


def test_manual_fit_cancel_restores_baseline_and_previous_unsaved_state():
    app_state = _state_with_overlays()
    app_state.unsaved_changes = False
    baseline = [(overlay.x, overlay.y, overlay.width, overlay.height) for overlay in app_state.overlays]
    session = ManualKeyboardFitSession(app_state)

    session.translate_group(10, 10)
    session.cancel()

    assert [(overlay.x, overlay.y, overlay.width, overlay.height) for overlay in app_state.overlays] == baseline
    assert app_state.unsaved_changes is False
