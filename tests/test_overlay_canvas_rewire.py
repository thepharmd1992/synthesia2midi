from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.canvas.interaction import CanvasInteraction
from synthesia2midi.gui.keyboard_canvas import KeyboardCanvas
from synthesia2midi.gui.overlay_interaction_controller import OverlayInteractionController
from synthesia2midi.gui.spark_calibration_controller import SparkCalibrationController
from synthesia2midi.workflows.overlay_manager import OverlayManager


class RecordingDisplayManager:
    def __init__(self):
        self.overlay_refreshes = 0

    def refresh_canvas_overlays(self):
        self.overlay_refreshes += 1


def _overlay(key_id=7):
    return OverlayConfig(
        key_id=key_id,
        note_octave=4,
        note_name_in_octave="C",
        x=10,
        y=20,
        width=30,
        height=40,
        key_type="LW",
    )


class _MouseEvent:
    def __init__(self, x, y, *, button=Qt.LeftButton, modifiers=Qt.NoModifier):
        self._x = x
        self._y = y
        self._button = button
        self._modifiers = modifiers

    def x(self):
        return self._x

    def y(self):
        return self._y

    def button(self):
        return self._button

    def modifiers(self):
        return self._modifiers


class _IdentityCoordManager:
    image_width = 200
    image_height = 120
    image_scale_factor = 1.0

    def image_rect_to_canvas(self, x, y, width, height):
        return x, y, width, height

    def scale_delta(self, dx, dy):
        return dx, dy

    def canvas_to_image(self, x, y, clamp_to_bounds=False):
        if clamp_to_bounds:
            x = max(0, min(self.image_width, x))
            y = max(0, min(self.image_height, y))
        return x, y


def test_overlay_manager_owns_index_based_move_and_resize_mutations():
    app_state = AppState()
    app_state.overlays = [_overlay()]
    manager = OverlayManager(app_state)

    assert manager.move_overlay_by_index(0, 15, 25) is True
    assert (app_state.overlays[0].x, app_state.overlays[0].y) == (15, 25)
    assert app_state.unsaved_changes is True

    app_state.unsaved_changes = False
    assert manager.resize_overlay_by_index(0, 1, 2, 3, 4) is True
    assert (
        app_state.overlays[0].x,
        app_state.overlays[0].y,
        app_state.overlays[0].width,
        app_state.overlays[0].height,
    ) == (1, 2, 3, 4)
    assert app_state.unsaved_changes is True

    app_state.unsaved_changes = False
    assert manager.move_overlay_by_index(99, 0, 0) is False
    assert manager.resize_overlay_by_index(99, 0, 0, 1, 1) is False
    assert app_state.unsaved_changes is False


def test_canvas_interaction_manual_fit_group_drag_emits_group_delta_not_single_move():
    app_state = AppState()
    overlays = [
        _overlay(key_id=1),
        _overlay(key_id=2),
    ]
    overlays[0].x = 0
    overlays[0].y = 20
    overlays[1].x = 40
    overlays[1].y = 20
    interaction = CanvasInteraction(None, _IdentityCoordManager(), app_state)
    interaction.set_callbacks(
        get_overlays=lambda: overlays,
        get_pixel_color=lambda x, y: None,
        get_current_frame=lambda: None,
    )
    interaction.set_manual_fit_mode("manual_fit_group")
    group_moves = []
    single_moves = []
    interaction.manual_fit_group_moved.connect(lambda dx, dy: group_moves.append((dx, dy)))
    interaction.overlay_moved.connect(lambda *args: single_moves.append(args))

    assert interaction.handle_mouse_press(_MouseEvent(35, 30)) is True
    assert interaction.handle_mouse_move(_MouseEvent(43, 34)) is True

    assert group_moves == [(8, 4)]
    assert single_moves == []


def test_canvas_interaction_manual_fit_single_mode_keeps_existing_overlay_drag():
    app_state = AppState()
    overlays = [_overlay(key_id=1)]
    interaction = CanvasInteraction(None, _IdentityCoordManager(), app_state)
    interaction.set_callbacks(
        get_overlays=lambda: overlays,
        get_pixel_color=lambda x, y: None,
        get_current_frame=lambda: None,
    )
    interaction.set_manual_fit_mode("manual_fit_single")
    group_moves = []
    single_moves = []
    interaction.manual_fit_group_moved.connect(lambda dx, dy: group_moves.append((dx, dy)))
    interaction.overlay_moved.connect(lambda *args: single_moves.append(args))

    assert interaction.handle_mouse_press(_MouseEvent(25, 40)) is True
    assert interaction.handle_mouse_move(_MouseEvent(30, 45)) is True

    assert group_moves == []
    assert single_moves == [(0, 15, 25)]


def test_canvas_interaction_manual_fit_region_mode_emits_vertical_range():
    app_state = AppState()
    interaction = CanvasInteraction(None, _IdentityCoordManager(), app_state)
    interaction.set_manual_fit_mode("manual_fit_black_region")
    regions = []
    interaction.manual_fit_region_selected.connect(
        lambda region_type, top, bottom: regions.append((region_type, top, bottom))
    )

    assert interaction.handle_mouse_press(_MouseEvent(10, 12)) is True
    assert interaction.handle_mouse_move(_MouseEvent(40, 42)) is True
    assert interaction.handle_mouse_release(_MouseEvent(40, 42)) is True

    assert regions == [("black", 12, 42)]
    assert interaction.manual_fit_mode() == "manual_fit_group"


def test_keyboard_canvas_overlay_handlers_delegate_geometry_to_overlay_manager():
    calls = []
    selected = []
    canvas = SimpleNamespace(
        _overlay_manager=SimpleNamespace(
            move_overlay_by_index=lambda *args: calls.append(("move", args)) or True,
            resize_overlay_by_index=lambda *args: calls.append(("resize", args)) or True,
        ),
        on_overlay_select_callback=lambda value: selected.append(value),
        update=lambda: calls.append(("update", ())),
    )

    KeyboardCanvas._handle_overlay_selected(canvas, 7)
    KeyboardCanvas._handle_overlay_selected(canvas, -1)
    KeyboardCanvas._handle_overlay_moved(canvas, 0, 11, 22)
    KeyboardCanvas._handle_overlay_resized(canvas, 0, 1, 2, 3, 4)

    assert selected == [7, None]
    assert ("move", (0, 11, 22)) in calls
    assert ("resize", (0, 1, 2, 3, 4)) in calls
    assert calls.count(("update", ())) == 2


def test_keyboard_canvas_resize_rebuilds_loaded_frame_pixmap(monkeypatch):
    QApplication.instance() or QApplication([])
    app_state = AppState()
    canvas = KeyboardCanvas(
        app_state,
        width=100,
        height=100,
        on_color_pick_callback=lambda *_: None,
        on_overlay_select_callback=lambda *_: None,
    )
    canvas.current_frame_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    canvas.coord_manager.update_image_size(200, 100)
    rebuilt = []
    monkeypatch.setattr(canvas, "_display_image", lambda: rebuilt.append(True))

    canvas.resizeEvent(QResizeEvent(QSize(120, 100), QSize(100, 100)))

    assert rebuilt == [True]


def test_canvas_interaction_empty_click_only_emits_selection_signal():
    app_state = AppState()
    app_state.ui.selected_overlay_id = 7
    interaction = CanvasInteraction(
        canvas_widget=None,
        coord_manager=SimpleNamespace(),
        app_state=app_state,
    )
    interaction.set_callbacks(
        get_overlays=lambda: [],
        get_pixel_color=lambda x, y: None,
        get_current_frame=lambda: None,
    )
    emitted = []
    interaction.overlay_selected.connect(lambda value: emitted.append(value))
    event = SimpleNamespace(x=lambda: 10, y=lambda: 10)

    assert interaction._handle_normal_press(event) is True

    assert emitted == [-1]
    assert app_state.ui.selected_overlay_id == 7


def test_keyboard_canvas_positions_white_labels_near_bottom_and_black_labels_centered():
    white_overlay = _overlay()
    white_overlay.note_name_in_octave = "C"
    black_overlay = _overlay()
    black_overlay.note_name_in_octave = "C♯"

    white_point = KeyboardCanvas._overlay_label_point(
        white_overlay,
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        text_width=20,
        text_height=12,
    )
    black_point = KeyboardCanvas._overlay_label_point(
        black_overlay,
        x1=0,
        y1=0,
        x2=100,
        y2=100,
        text_width=20,
        text_height=12,
    )

    assert white_point.x() == 40
    assert white_point.y() == 83
    assert black_point.x() == 40
    assert black_point.y() == 53


def test_keyboard_canvas_uses_readable_label_color_pair_for_overlay_colors():
    dark_text, dark_shadow = KeyboardCanvas._overlay_label_colors("red")
    bright_text, bright_shadow = KeyboardCanvas._overlay_label_colors("yellow")

    assert dark_text.name() == "#ffffff"
    assert dark_shadow.name() == "#000000"
    assert bright_text.name() == "#000000"
    assert bright_shadow.name() == "#ffffff"


def test_keyboard_canvas_masks_white_border_segments_inside_black_overlay_rects():
    white_rect = QRect(0, 0, 100, 100)
    black_rect = QRect(70, 0, 60, 55)

    segments = KeyboardCanvas._overlay_border_segments(
        white_rect,
        blocking_rects=[black_rect],
    )

    assert ((99, 0), (99, 54)) not in segments
    assert ((99, 55), (99, 99)) in segments
    assert ((0, 0), (69, 0)) in segments
    assert ((70, 0), (99, 0)) not in segments


def test_spark_roi_update_controller_owns_state_cache_refresh_and_ui(monkeypatch):
    app_state = AppState()
    display_manager = RecordingDisplayManager()
    control_panel = SimpleNamespace(updated=0)
    control_panel.update_controls_from_state = lambda: setattr(control_panel, "updated", control_panel.updated + 1)
    invalidations = []

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    from synthesia2midi.detection import spark_mapper

    monkeypatch.setattr(
        spark_mapper,
        "get_spark_mapper",
        lambda: SimpleNamespace(invalidate_cache=lambda: invalidations.append("invalidated")),
    )

    app = SimpleNamespace(
        app_state=app_state,
        display_manager=display_manager,
        control_panel=control_panel,
    )

    SparkCalibrationController(app)._handle_spark_roi_updated(12, 34)

    assert app_state.detection.spark_roi_top == 12
    assert app_state.detection.spark_roi_bottom == 34
    assert app_state.detection.spark_roi_visible is True
    assert app_state.unsaved_changes is True
    assert invalidations == ["invalidated"]
    assert display_manager.overlay_refreshes == 1
    assert control_panel.updated == 1


def test_overlay_type_change_uses_interaction_api_and_display_manager_refresh():
    app_state = AppState()
    display_manager = RecordingDisplayManager()
    interaction_calls = []
    interaction = SimpleNamespace(
        set_overlay_drawing_type=lambda value: interaction_calls.append(value),
    )
    canvas = SimpleNamespace(
        interaction=interaction,
        draw_overlays=lambda: pytest.fail("controller should refresh through DisplayManager"),
    )
    app = SimpleNamespace(
        app_state=app_state,
        display_manager=display_manager,
        keyboard_canvas=canvas,
    )

    OverlayInteractionController(app)._handle_overlay_type_change("spark")

    assert app_state.ui.overlay_drawing_type == "spark"
    assert interaction_calls == ["spark"]
    assert display_manager.overlay_refreshes == 1
