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
    assert [(o.x, o.y) for o in app_state.overlays] == [
        (7, 17),
        (19, 7),
        (31, 17),
    ]
    assert app_state.unsaved_changes is True


def test_manual_fit_keyboard_width_scales_white_and_black_keys_from_same_span():
    app_state = _state_with_overlays()
    baseline_span = _bounds(app_state.overlays)[2]
    session = ManualKeyboardFitSession(app_state)

    session.set_param("keyboard_width_delta", baseline_span)

    assert _bounds(app_state.overlays)[2] == pytest.approx(baseline_span * 2)
    assert app_state.overlays[0].width == pytest.approx(20)
    assert app_state.overlays[1].width == pytest.approx(12)
    assert app_state.overlays[2].width == pytest.approx(20)


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


def test_manual_fit_remaining_shape_controls_are_color_scoped():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.update_params(
        ManualFitParams(
            white_height_delta=6,
            black_y_delta=-3,
            black_height_delta=5,
            black_width_delta=4,
        )
    )

    white_left, black, white_right = app_state.overlays
    assert (white_left.y, white_left.height, white_left.width) == pytest.approx((20, 46, 10))
    assert (white_right.y, white_right.height, white_right.width) == pytest.approx((20, 46, 10))
    assert (black.y, black.height, black.width) == pytest.approx((7, 25, 10))
    assert black.x == pytest.approx(10)


def test_manual_fit_removed_geometry_controls_are_not_backend_parameters():
    removed_names = {"white_y_delta", "white_width_delta", "black_x_delta"}

    assert removed_names.isdisjoint({field.name for field in fields(ManualFitParams)})


def test_manual_fit_single_overlay_override_survives_later_group_changes():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.move_single_overlay_by_index(0, 100, 50)
    session.translate_group(10, 0)

    assert app_state.overlays[0].x == pytest.approx(110)
    assert app_state.overlays[0].y == pytest.approx(50)
    assert app_state.overlays[1].x == pytest.approx(22)
    assert session.overridden_key_ids() == {1}


def test_manual_fit_control_updates_preserve_current_group_position():
    app_state = _state_with_overlays()
    session = ManualKeyboardFitSession(app_state)

    session.translate_group(30, 8)
    session.update_control_params(ManualFitParams(keyboard_width_delta=20))

    assert session.params.group_dx == pytest.approx(30)
    assert session.params.group_dy == pytest.approx(8)
    assert app_state.overlays[0].x > 0
    assert app_state.overlays[0].y == pytest.approx(28)


def test_manual_fit_cancel_restores_baseline_and_previous_unsaved_state():
    app_state = _state_with_overlays()
    app_state.unsaved_changes = False
    baseline = [(overlay.x, overlay.y, overlay.width, overlay.height) for overlay in app_state.overlays]
    session = ManualKeyboardFitSession(app_state)

    session.translate_group(10, 10)
    session.cancel()

    assert [(overlay.x, overlay.y, overlay.width, overlay.height) for overlay in app_state.overlays] == baseline
    assert app_state.unsaved_changes is False
