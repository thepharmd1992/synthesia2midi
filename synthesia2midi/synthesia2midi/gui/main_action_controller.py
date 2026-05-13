"""Small UI action handlers that mutate app state or delegate to managers."""
from __future__ import annotations

import logging


class MainActionController:
    """Owns remaining menu/control action handlers that do not belong in the root window."""

    def __init__(self, app):
        self.app = app

    def toggle_overlays(self) -> None:
        if self.app.display_manager:
            self.app.display_manager.toggle_overlays()

    def toggle_live_detection_feedback(self) -> None:
        if self.app.display_manager:
            self.app.display_manager.toggle_live_detection_feedback()

    def handle_visual_threshold_monitor_menu(self, checked: bool) -> None:
        app = self.app
        app.app_state.ui.visual_threshold_monitor_enabled = checked
        app.app_state.unsaved_changes = True
        app.visual_threshold_monitor_action.setChecked(checked)
        if app.display_manager:
            app.display_manager.handle_visual_threshold_monitor_toggle(checked)
        logging.info("Visual threshold monitor: %s", "enabled" if checked else "disabled")

    def handle_calibrate_unlit_all_keys(self) -> None:
        if self.app.calibration_workflow:
            self.app.calibration_workflow.handle_calibrate_unlit_all_keys()

    def handle_calibrate_lit_exemplar_key_start(self, key_type: str) -> None:
        if self.app.calibration_workflow:
            self.app.calibration_workflow.handle_calibrate_lit_exemplar_key_start(key_type)

    def handle_detection_threshold_change(self, threshold: float) -> None:
        if self.app.detection_manager:
            self.app.detection_manager.handle_detection_threshold_change(threshold)

    def handle_rise_delta_threshold_change(self, threshold: float) -> None:
        self.app.app_state.detection.rise_delta_threshold = threshold
        self.app.app_state.unsaved_changes = True

    def handle_fall_delta_threshold_change(self, threshold: float) -> None:
        self.app.app_state.detection.fall_delta_threshold = threshold
        self.app.app_state.unsaved_changes = True

    def handle_histogram_threshold_change(self, threshold: float) -> None:
        self.app.app_state.detection.hist_ratio_threshold = threshold
        self.app.app_state.unsaved_changes = True

    def handle_similarity_ratio_change(self, ratio: float) -> None:
        self.app.app_state.detection.similarity_ratio = ratio
        self.app.app_state.unsaved_changes = True

    def handle_refresh_selected_overlay_display(self) -> None:
        if self.app.display_manager:
            self.app.display_manager.handle_refresh_selected_overlay_display()

    def handle_align_white_keys_to_selected(self) -> None:
        if self.app.overlay_manager:
            self.app.overlay_manager.handle_align_white_keys_to_selected()

    def handle_align_black_keys_to_selected(self) -> None:
        if self.app.overlay_manager:
            self.app.overlay_manager.handle_align_black_keys_to_selected()

    def handle_overlay_size_adjustment(self, key_color: str, dimension: str, delta: int) -> None:
        self.app.overlay_manager.adjust_overlay_sizes(key_color, dimension, delta)

    def toggle_hist_detection(self) -> None:
        if self.app.detection_manager:
            self.app.detection_manager.toggle_histogram_detection()

    def toggle_delta_detection(self) -> None:
        if self.app.detection_manager:
            self.app.detection_manager.toggle_delta_detection()

    def toggle_winner_takes_black(self, enabled: bool) -> None:
        self.app.app_state.detection.winner_takes_black_enabled = enabled
        self.app.app_state.unsaved_changes = True
        logging.info("Black key filter (winner takes black) is now %s", enabled)

    def handle_exemplar_key_type_enabled_change(self, key_type: str, enabled: bool) -> None:
        if key_type not in {"LW", "LB", "RW", "RB"}:
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

    def handle_hand_assignment_toggle(self, enabled: bool) -> None:
        self.app.app_state.detection.hand_assignment_enabled = enabled
        self.app.app_state.unsaved_changes = True
        logging.info("Hand assignment is now %s", enabled)

    def handle_overlay_color_change(self, color: str) -> None:
        app = self.app
        logging.debug("Overlay color changed to: %s", color)
        app.app_state.ui.overlay_color = color.lower()
        if app.keyboard_canvas:
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
        if app.keyboard_canvas:
            app.keyboard_canvas.draw_overlays()
        app.app_state.mark_unsaved()

    def resize_and_position_window(self) -> None:
        self.app.window_manager.resize_and_position_window()

    def create_detection_wrapper(self):
        if self.app.detection_manager:
            return self.app.detection_manager.create_detection_wrapper()
        return None
