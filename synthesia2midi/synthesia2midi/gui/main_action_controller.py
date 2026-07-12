"""Small UI action handlers that mutate app state or delegate to managers."""
from __future__ import annotations

import logging

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.core.color_families import (
    SUPPORTED_EXEMPLAR_SLOTS,
    active_family_numbers,
    slots_for_family,
)


class MainActionController:
    """Owns remaining menu/control action handlers that do not belong in the root window."""

    def __init__(self, app):
        self.app = app

    def _detection_manager(self):
        return getattr(self.app, "detection_manager", None)

    def toggle_overlays(self) -> None:
        if self.app.display_manager:
            self.app.display_manager.toggle_overlays()

    def toggle_live_detection_feedback(self, enabled=None) -> None:
        display_manager = getattr(self.app, "display_manager", None)
        if display_manager:
            display_manager.toggle_live_detection_feedback(enabled)

    def handle_visual_threshold_monitor_menu(self, checked: bool) -> None:
        display_manager = getattr(self.app, "display_manager", None)
        if display_manager:
            display_manager.set_visual_threshold_monitor_enabled(checked)

    def handle_calibrate_unlit_all_keys(self) -> None:
        if self.app.calibration_workflow:
            self.app.calibration_workflow.handle_calibrate_unlit_all_keys()

    def handle_calibrate_lit_exemplar_key_start(self, key_type: str) -> None:
        if self.app.calibration_workflow:
            self.app.calibration_workflow.handle_calibrate_lit_exemplar_key_start(key_type)

    def handle_detection_threshold_change(self, threshold: float) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_detection_threshold(threshold)

    def handle_rise_delta_threshold_change(self, threshold: float) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_rise_delta_threshold(threshold)

    def handle_fall_delta_threshold_change(self, threshold: float) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_fall_delta_threshold(threshold)

    def handle_histogram_threshold_change(self, threshold: float) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_histogram_threshold(threshold)

    def handle_similarity_ratio_change(self, ratio: float) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_similarity_ratio(ratio)

    def handle_refresh_selected_overlay_display(self) -> None:
        if self.app.display_manager:
            self.app.display_manager.handle_refresh_selected_overlay_display()

    def handle_align_white_keys_to_selected(self) -> None:
        if self.app.overlay_manager:
            self.app.overlay_manager.handle_align_white_keys_to_selected()

    def handle_align_black_keys_to_selected(self) -> None:
        if self.app.overlay_manager:
            self.app.overlay_manager.handle_align_black_keys_to_selected()

    def handle_manual_fit_request(self) -> None:
        controller = getattr(self.app, "manual_keyboard_fit_controller", None)
        if controller is not None:
            controller.open()

    def handle_overlay_size_adjustment(self, key_color: str, dimension: str, delta: int) -> None:
        result = self.app.overlay_manager.adjust_overlay_sizes(key_color, dimension, delta)
        control_panel = getattr(self.app, "control_panel", None)
        if result is not None and control_panel is not None and hasattr(control_panel, "apply_overlay_adjustment_result"):
            control_panel.apply_overlay_adjustment_result(result)

    def toggle_hist_detection(self, enabled=None) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.toggle_histogram_detection(enabled)

    def toggle_delta_detection(self, enabled=None) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.toggle_delta_detection(enabled)

    def toggle_winner_takes_black(self, enabled: bool) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_winner_takes_black_enabled(enabled)

    def handle_exemplar_key_type_enabled_change(self, key_type: str, enabled: bool) -> None:
        if key_type not in SUPPORTED_EXEMPLAR_SLOTS:
            logging.warning("Ignoring invalid exemplar key type toggle: %s", key_type)
            return

        app = self.app
        app.app_state.detection.exemplar_key_type_enabled[key_type] = enabled
        app.app_state.unsaved_changes = True
        logging.info("Exemplar key type %s availability set to %s", key_type, enabled)

        if (
            not enabled
            and app.app_state.calibration.calibration_mode == "lit_exemplar"
            and app.app_state.calibration.current_calibration_key_type == key_type
        ):
            app.app_state.calibration.calibration_mode = None
            app.app_state.calibration.current_calibration_key_type = None
            logging.info("Cancelled lit exemplar calibration for disabled key type %s", key_type)

        if app.control_panel:
            app.control_panel.update_controls_from_state()

    def handle_add_additional_color(self) -> None:
        detection = self.app.app_state.detection
        active_families = set(
            active_family_numbers(
                detection.exemplar_key_type_enabled,
                detection.exemplar_lit_colors,
            )
        )
        family_number = next(
            (number for number in range(2, 5) if number not in active_families),
            None,
        )
        if family_number is None:
            return

        for slot in slots_for_family(family_number):
            detection.exemplar_key_type_enabled[slot] = True
        if family_number >= 3:
            detection.hand_assignment_enabled = True
        self.app.app_state.unsaved_changes = True
        if self.app.control_panel:
            self.app.control_panel.update_controls_from_state()

    def handle_remove_additional_color(self, family_number: int) -> None:
        if family_number not in {2, 3, 4}:
            return

        app = self.app
        detection = app.app_state.detection
        slots = slots_for_family(family_number)
        colors = detection.exemplar_lit_colors
        histograms = detection.exemplar_lit_histograms
        enabled = detection.exemplar_key_type_enabled
        has_saved_data = any(
            colors.get(slot) is not None or histograms.get(slot) is not None
            for slot in slots
        )
        if has_saved_data:
            response = QMessageBox.question(
                getattr(app, "control_panel", None),
                QCoreApplication.translate(
                    "MainActionController", "Remove Color {number}"
                ).format(number=family_number),
                QCoreApplication.translate(
                    "MainActionController",
                    "Remove Color {number} and delete its saved calibration data?",
                ).format(number=family_number),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if response != QMessageBox.Yes:
                return

        if not has_saved_data and not any(enabled.get(slot, False) for slot in slots):
            return

        for slot in slots:
            enabled[slot] = False
            colors[slot] = None
            histograms[slot] = None

        calibration = app.app_state.calibration
        if (
            calibration.calibration_mode == "lit_exemplar"
            and calibration.current_calibration_key_type in slots
        ):
            calibration.calibration_mode = None
            calibration.current_calibration_key_type = None

        app.app_state.unsaved_changes = True
        if app.control_panel:
            app.control_panel.update_controls_from_state()

    def handle_hand_assignment_toggle(self, enabled: bool) -> None:
        detection_manager = self._detection_manager()
        if detection_manager:
            detection_manager.set_hand_assignment_enabled(enabled)

    def handle_overlay_color_change(self, color: str) -> None:
        app = self.app
        logging.debug("Overlay color changed to: %s", color)
        app.app_state.ui.overlay_color = color.lower()
        display_manager = getattr(app, "display_manager", None)
        if display_manager:
            display_manager.refresh_canvas_overlays()
        elif getattr(app, "keyboard_canvas", None):
            app.keyboard_canvas.update()
        app.app_state.unsaved_changes = True

    def handle_fps_override_change(self, fps_override) -> None:
        app = self.app
        logging.info("Setting FPS override to: %s", fps_override)
        app.app_state.video.fps_override = fps_override
        app.app_state.unsaved_changes = True
        if app.video_session:
            if fps_override:
                logging.info("FPS override set to %s (detected: %s)", fps_override, app.video_session.fps)
            else:
                logging.info("FPS override disabled, using detected: %s", app.video_session.fps)
            app.control_panel.update_video_info(app.video_session.fps)

    def handle_octave_transpose_change(self, transpose_value: int) -> None:
        app = self.app
        logging.info("Applying octave transpose: %s", transpose_value)
        app.app_state.midi.octave_transpose = transpose_value
        display_manager = getattr(app, "display_manager", None)
        if display_manager:
            display_manager.refresh_canvas_overlays()
        elif getattr(app, "keyboard_canvas", None):
            app.keyboard_canvas.draw_overlays()
        app.app_state.mark_unsaved()

    def resize_and_position_window(self) -> None:
        self.app.window_manager.resize_and_position_window()

    def create_detection_wrapper(self):
        detection_manager = self._detection_manager()
        if detection_manager:
            return detection_manager.create_detection_wrapper()
        return None
