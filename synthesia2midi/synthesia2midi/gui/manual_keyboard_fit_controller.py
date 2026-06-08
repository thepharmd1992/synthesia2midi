"""Controller for the Manual Keyboard Fit tool."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.gui.dialog_positioning import move_to_top_center_safe_zone
from synthesia2midi.gui.manual_keyboard_fit_dialog import ManualKeyboardFitDialog
from synthesia2midi.workflows.manual_keyboard_fit import ManualFitParams, ManualKeyboardFitSession


class ManualKeyboardFitController:
    """Owns Manual Fit dialog lifecycle and canvas interaction routing."""

    def __init__(
        self,
        app,
        *,
        dialog_factory: Optional[Callable[..., ManualKeyboardFitDialog]] = None,
    ) -> None:
        self.app = app
        self._dialog_factory = dialog_factory or ManualKeyboardFitDialog
        self._dialog: Optional[ManualKeyboardFitDialog] = None
        self._session: Optional[ManualKeyboardFitSession] = None
        self._settings_tool_was_visible = False
        self._finishing = False
        self._setup_step: Optional[str] = None
        self._manual_fit_overlays_visible = True

    @property
    def active_dialog(self) -> Optional[ManualKeyboardFitDialog]:
        return self._dialog

    @property
    def session(self) -> Optional[ManualKeyboardFitSession]:
        return self._session

    def open(self, *, start_setup: bool = False) -> bool:
        if not self.app.app_state.overlays:
            QMessageBox.warning(
                self.app,
                "Manual Fit",
                "Generate overlays before opening Manual Fit.",
            )
            return False

        if self._dialog is not None:
            self._dialog.raise_()
            self._dialog.activateWindow()
            return True

        self._session = ManualKeyboardFitSession(self.app.app_state)
        self._settings_tool_was_visible = self._hide_settings_tool_window()

        self.app.app_state.ui.show_overlays = True
        if hasattr(self.app, "show_overlays_action"):
            self.app.show_overlays_action.setChecked(True)

        self.app.keyboard_canvas.set_manual_fit_callbacks(
            group_move_callback=self._handle_group_move,
            single_move_callback=self._handle_single_move,
            single_resize_callback=self._handle_single_resize,
            override_ids_callback=self._override_ids,
            region_selected_callback=self._handle_region_selected,
            region_guides_callback=self._region_guides,
            keyboard_box_selected_callback=self._handle_keyboard_box_selected,
            guide_line_changed_callback=self._handle_guide_line_changed,
            guide_line_selected_callback=self._handle_guide_line_selected,
            setup_instruction_callback=self._setup_instruction,
            overlays_visible_callback=self._overlays_visible,
        )

        dialog = self._dialog_factory(
            self.app,
            initial_octave=self.app.app_state.midi.octave_transpose,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.params_changed.connect(self._handle_params_changed)
        dialog.octave_changed.connect(self._handle_octave_changed)
        dialog.mode_changed.connect(self._handle_mode_changed)
        dialog.setup_back_requested.connect(self._handle_setup_back)
        dialog.setup_use_suggested_requested.connect(self._handle_setup_use_suggested)
        dialog.reset_all_requested.connect(self._handle_reset_all)
        dialog.reset_position_requested.connect(self._handle_reset_position)
        dialog.clear_selected_override_requested.connect(self._handle_clear_selected_override)
        dialog.accepted.connect(self._handle_apply)
        dialog.rejected.connect(self._handle_cancel)
        self._dialog = dialog

        if start_setup:
            self._start_setup()
        else:
            self._finish_setup()

        move_to_top_center_safe_zone(dialog, self.app)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def _handle_params_changed(self, params: ManualFitParams) -> None:
        if self._session is None:
            return
        self._session.update_control_params(params)
        self._refresh_preview()

    def _handle_octave_changed(self, value: int) -> None:
        if self._session is None:
            return
        self._session.set_octave_transpose(value)
        self._sync_octave_control()
        self._refresh_preview()

    def _handle_mode_changed(self, mode: str) -> None:
        self.app.keyboard_canvas.set_manual_fit_mode(mode)

    def _handle_group_move(self, dx: float, dy: float) -> None:
        if self._session is None:
            return
        self._session.translate_group(dx, dy)
        self._refresh_preview()

    def _handle_single_move(self, overlay_index: int, new_x: float, new_y: float) -> bool:
        if self._session is None:
            return False
        updated = self._session.move_single_overlay_by_index(overlay_index, new_x, new_y)
        self._refresh_preview()
        return updated

    def _handle_region_selected(self, region_type: str, top: float, bottom: float) -> None:
        if self._session is None:
            return
        self._session.set_detection_region(region_type, top, bottom)
        self.app.keyboard_canvas.set_manual_fit_mode("manual_fit_group")
        self._refresh_preview()

    def _handle_keyboard_box_selected(self, left: float, top: float, right: float, bottom: float) -> None:
        if self._session is None or self._setup_step != "keyboard_box":
            return
        self._session.set_setup_keyboard_box(left, top, right, bottom)
        self._enter_setup_step("black_bottom")

    def _handle_guide_line_changed(self, line_type: str, y: float) -> None:
        if self._session is None:
            return
        if line_type == "black_bottom" and self._setup_step == "black_bottom":
            self._session.set_setup_black_bottom(y)
        elif line_type == "white_start" and self._setup_step == "white_start":
            self._session.set_setup_white_start(y)
        else:
            return
        self._refresh_preview()

    def _handle_guide_line_selected(self, line_type: str, y: float) -> None:
        self._handle_guide_line_changed(line_type, y)
        if self._session is None:
            return
        if line_type == "black_bottom" and self._setup_step == "black_bottom":
            self._enter_setup_step("white_start")
            return
        if line_type == "white_start" and self._setup_step == "white_start":
            self._finish_setup_from_session()

    def _handle_setup_back(self) -> None:
        if self._setup_step == "black_bottom":
            self._enter_setup_step("keyboard_box")
        elif self._setup_step == "white_start":
            self._enter_setup_step("black_bottom")

    def _handle_setup_use_suggested(self) -> None:
        if self._session is None:
            return
        if self._setup_step == "black_bottom":
            self._enter_setup_step("white_start")
        elif self._setup_step == "white_start":
            self._finish_setup_from_session()

    def _handle_single_resize(
        self,
        overlay_index: int,
        new_x: float,
        new_y: float,
        new_width: float,
        new_height: float,
    ) -> bool:
        if self._session is None:
            return False
        updated = self._session.resize_single_overlay_by_index(
            overlay_index,
            new_x,
            new_y,
            new_width,
            new_height,
        )
        self._refresh_preview()
        return updated

    def _handle_reset_all(self) -> None:
        if self._session is None:
            return
        self._session.reset_all()
        if self._dialog is not None:
            self._dialog.reset_controls(octave_value=self._session.current_octave_transpose())
        self._sync_octave_control()
        self._refresh_preview()

    def _handle_reset_position(self) -> None:
        if self._session is None:
            return
        self._session.reset_position()
        self._refresh_preview()

    def _handle_clear_selected_override(self) -> None:
        if self._session is None:
            return
        self._session.clear_selected_override()
        self._refresh_preview()

    def _handle_apply(self) -> None:
        if self._session is not None:
            self._session.apply()
        self._finish()

    def _handle_cancel(self) -> None:
        if self._session is not None:
            self._session.cancel()
        self._finish()

    def _finish(self) -> None:
        if self._finishing:
            return
        self._finishing = True
        dialog = self._dialog
        try:
            self._setup_step = None
            self._manual_fit_overlays_visible = True
            self.app.keyboard_canvas.clear_manual_fit_callbacks()
            self._restore_settings_tool_window()
            self._sync_octave_control()
            self._refresh_preview()
            self._dialog = None
            self._session = None
            if dialog is not None:
                dialog.deleteLater()
        finally:
            self._finishing = False

    def _override_ids(self) -> set[int]:
        if self._session is None:
            return set()
        return self._session.overridden_key_ids()

    def _region_guides(self) -> dict:
        if self._session is None:
            return {}
        if self._setup_step is not None:
            return self._session.setup_guides_for_step(self._setup_step)
        return self._session.detection_region_guides()

    def _overlays_visible(self) -> bool:
        return self._manual_fit_overlays_visible

    def _setup_instruction(self) -> str:
        instructions = {
            "keyboard_box": "Draw a box around the visible keyboard",
            "black_bottom": "Drag to bottom of black keys",
            "white_start": "Drag to start of white-key detection area",
        }
        return instructions.get(self._setup_step or "", "")

    def _start_setup(self) -> None:
        self._manual_fit_overlays_visible = False
        self._enter_setup_step("keyboard_box")

    def _enter_setup_step(self, step: str) -> None:
        self._setup_step = step
        mode_by_step = {
            "keyboard_box": "manual_fit_keyboard_box",
            "black_bottom": "manual_fit_black_bottom",
            "white_start": "manual_fit_white_start",
        }
        self.app.keyboard_canvas.set_manual_fit_mode(mode_by_step[step])
        if self._dialog is not None:
            self._dialog.enter_setup_step(step)
        self._refresh_preview()

    def _finish_setup_from_session(self) -> None:
        if self._session is None:
            return
        self._session.finalize_setup_geometry()
        self._finish_setup()
        self._refresh_preview()

    def _finish_setup(self) -> None:
        self._setup_step = None
        self._manual_fit_overlays_visible = True
        self.app.keyboard_canvas.set_manual_fit_mode("manual_fit_group")
        if self._dialog is not None:
            self._dialog.finish_setup()
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if hasattr(self.app.keyboard_canvas, "update"):
            self.app.keyboard_canvas.update()
        control_panel = getattr(self.app, "control_panel", None)
        if control_panel is not None and hasattr(control_panel, "update_selected_overlay_display"):
            control_panel.update_selected_overlay_display()

    def _sync_octave_control(self) -> None:
        control_panel = getattr(self.app, "control_panel", None)
        if control_panel is None:
            return
        octave_spin = getattr(control_panel, "octave_transpose_spin", None)
        if octave_spin is not None:
            with QSignalBlocker(octave_spin):
                octave_spin.setValue(self.app.app_state.midi.octave_transpose)
            return
        if hasattr(control_panel, "update_controls_from_state"):
            control_panel.update_controls_from_state()

    def _hide_settings_tool_window(self) -> bool:
        settings_tool_window = getattr(self.app, "settings_tool_window", None)
        if settings_tool_window is None or not settings_tool_window.isVisible():
            return False
        settings_tool_window.hide()
        return True

    def _restore_settings_tool_window(self) -> None:
        if not self._settings_tool_was_visible:
            return
        self._settings_tool_was_visible = False
        settings_tool_window = getattr(self.app, "settings_tool_window", None)
        if settings_tool_window is None:
            return
        if hasattr(settings_tool_window, "show_preserving_geometry"):
            settings_tool_window.show_preserving_geometry()
        else:
            settings_tool_window.show_near_parent()
