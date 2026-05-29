"""Overlay-click calibration interaction controller."""
from __future__ import annotations

import logging
import traceback
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox


class CalibrationInteractionController:
    """Owns overlay click dispatch for calibration modes."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    def _handle_color_pick(self, color_rgb: Tuple[int, int, int], coordinates: Tuple[int, int]):
        """KeyboardCanvas callback adapter; keep this named slot for interaction wiring."""
        if self.calibration_workflow:
            self.calibration_workflow.handle_color_pick(color_rgb, coordinates)

    def _handle_overlay_selection(self, selected_key_id: Optional[int]):
        self.app_state.ui.selected_overlay_id = selected_key_id
        logging.debug(f"Overlay selected: key_id {selected_key_id}")

        if self.app_state.calibration.calibration_mode == "lit_exemplar" and \
           self.app_state.calibration.current_calibration_key_type and \
           selected_key_id is not None:

            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample and self.keyboard_canvas:
                sampled_color = self.keyboard_canvas.get_average_color_for_overlay(self.keyboard_canvas.current_frame_rgb, overlay_to_sample)
                if sampled_color:
                    key_type_to_cal = self.app_state.calibration.current_calibration_key_type
                    if (
                        key_type_to_cal in {"LW", "LB", "RW", "RB"}
                        and not self.app_state.detection.exemplar_key_type_enabled.get(key_type_to_cal, True)
                    ):
                        logging.info(f"Skipping exemplar calibration for disabled key type {key_type_to_cal}")
                        QMessageBox.warning(
                            self.app,
                            "Calibration Disabled",
                            f"{key_type_to_cal} is marked as not present in this video. Re-enable 'Present in Video' first."
                        )
                        self.app_state.calibration.calibration_mode = None
                        self.app_state.calibration.current_calibration_key_type = None
                        self.control_panel.update_controls_from_state()
                        return
                    self.app_state.detection.exemplar_lit_colors[key_type_to_cal] = sampled_color
                    logging.info(f"Calibrated exemplar lit color for {key_type_to_cal} to {sampled_color} from overlay {selected_key_id}.")

                    # Also capture histogram from the same overlay
                    if self.keyboard_canvas.current_frame_rgb is not None:
                        from synthesia2midi.detection.roi_utils import get_hist_feature
                        try:
                            roi_bgr = self.keyboard_canvas.get_roi_bgr(overlay_to_sample)
                            if roi_bgr is not None and roi_bgr.size > 0:
                                exemplar_hist = get_hist_feature(roi_bgr)
                                if exemplar_hist is not None:
                                    self.app_state.detection.exemplar_lit_histograms[key_type_to_cal] = exemplar_hist
                                    logging.info(f"Captured exemplar lit histogram for {key_type_to_cal} from overlay {selected_key_id}: shape={exemplar_hist.shape}, sum={exemplar_hist.sum()}")
                                else:
                                    logging.warning(f"get_hist_feature returned None for {key_type_to_cal}")

                                # Extract hue for hand detection calibration (only if hand assignment is enabled)
                                if self.app_state.detection.hand_assignment_enabled:
                                    logging.debug(f"[HUE-CALIBRATION] Extracting hue for {key_type_to_cal} from overlay {selected_key_id}")
                                    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                                    avg_hue = np.mean(hsv[:, :, 0])
                                    logging.debug(f"[HUE-CALIBRATION] Average hue extracted: {avg_hue}")

                                    # Update hand detection calibration based on key type
                                    if key_type_to_cal.startswith('L'):  # Left hand
                                        logging.debug(f"[HUE-CALIBRATION] Updating left_hand_hue_mean from {self.app_state.detection.left_hand_hue_mean} to {avg_hue}")
                                        self.app_state.detection.left_hand_hue_mean = avg_hue
                                    elif key_type_to_cal.startswith('R'):  # Right hand
                                        logging.debug(f"[HUE-CALIBRATION] Updating right_hand_hue_mean from {self.app_state.detection.right_hand_hue_mean} to {avg_hue}")
                                        self.app_state.detection.right_hand_hue_mean = avg_hue
                                    else:
                                        logging.error(f"[HUE-CALIBRATION] ERROR: Invalid key type: {key_type_to_cal}")

                                    # If both hands are calibrated, mark as calibrated
                                    if self.app_state.detection.left_hand_hue_mean > 0 and self.app_state.detection.right_hand_hue_mean > 0:
                                        logging.info(f"[HUE-CALIBRATION] Both hands calibrated. Marking hand_detection_calibrated = True")
                                        self.app_state.detection.hand_detection_calibrated = True

                                    logging.debug(f"[HUE-CALIBRATION] Current hand detection state:")
                                    logging.debug(f"[HUE-CALIBRATION]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
                                    logging.debug(f"[HUE-CALIBRATION]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
                                    logging.debug(f"[HUE-CALIBRATION]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
                                else:
                                    logging.debug(f"[HUE-CALIBRATION] Hand assignment disabled - skipping hue extraction")
                            else:
                                logging.warning(f"Could not extract valid ROI BGR from overlay {selected_key_id} for {key_type_to_cal}")
                        except Exception as e:
                            logging.error(f"Error capturing exemplar histogram for {key_type_to_cal}: {e}")
                            logging.error(f"Traceback: {traceback.format_exc()}")

                    # Auto-save calibration changes
                    if hasattr(self, 'video_loading_workflow') and self.video_loading_workflow:
                        success = self.video_loading_workflow.save_current_config()
                        if success:
                            logging.info("Exemplar calibration changes automatically saved to config file.")
                            save_msg = "\\nCalibration data automatically saved."
                        else:
                            logging.warning("Auto-save of exemplar calibration changes failed.")
                            save_msg = "\\nWarning: Auto-save failed."
                    else:
                        save_msg = "\\nWarning: Auto-save not available."

                    # Update GUI display before resetting calibration mode
                    self.control_panel.update_advanced_calibration_display()
                    self.control_panel.update_controls_from_state()  # Update color swatches

                    # Check if histogram was captured for user feedback
                    hist_captured = "Yes" if self.app_state.detection.exemplar_lit_histograms.get(key_type_to_cal) is not None else "No"

                    # Reset calibration state after successful completion
                    self.app_state.calibration.calibration_mode = None
                    self.app_state.calibration.current_calibration_key_type = None
                    self.app_state.unsaved_changes = False  # Reset since we auto-saved

                    # Simple success message
                    QMessageBox.information(self.app, "Calibration Successful",
                                          f"Calibration of {key_type_to_cal} successful")
                else:
                    QMessageBox.warning(self.app, "Calibration Error", "Could not sample color from the selected overlay.")
            # Always reset calibration mode after an attempt
            self.app_state.calibration.calibration_mode = None
            self.app_state.calibration.current_calibration_key_type = None

        # Handle spark calibration modes
        elif self.app_state.calibration.calibration_mode in ["spark_bar_only", "spark_dimmest_sparks",
                                                              "spark_lh_bar_only", "spark_lh_dimmest_sparks",
                                                              "spark_lh_brightest_sparks",
                                                              "spark_rh_bar_only", "spark_rh_dimmest_sparks",
                                                              "spark_rh_brightest_sparks"] and \
             selected_key_id is not None:

            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample:
                self.app.calibration_effects_controller.spark.capture_spark_overlay_calibration(
                    overlay_to_sample,
                    self.app_state.calibration.calibration_mode,
                )
            else:
                QMessageBox.warning(self.app, "Calibration Error", f"Could not find overlay for key {selected_key_id}.")
                self.app_state.calibration.calibration_mode = None

        # Handle shadow calibration modes
        elif (self.app_state.calibration.calibration_mode and
              self.app_state.calibration.calibration_mode.startswith("shadow_") and
              selected_key_id is not None):

            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample:
                self.app.calibration_effects_controller.shadow.capture_shadow_overlay_calibration(
                    overlay_to_sample,
                    self.app_state.calibration.calibration_mode,
                )
            else:
                QMessageBox.warning(self.app, "Calibration Error", f"Could not find overlay for key {selected_key_id}.")
                self.app_state.calibration.calibration_mode = None

        # Handle auto-calibration modes
        elif (self.app_state.calibration.calibration_mode and
              self.app_state.calibration.calibration_mode.startswith("auto_calibrate_") and
              selected_key_id is not None):

            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            if overlay_to_sample and self.auto_calibration_workflow:
                # Get ROI from the selected overlay
                roi_bgr = self.keyboard_canvas.get_roi_bgr(overlay_to_sample)
                if roi_bgr is not None:
                    # Handle the overlay click through auto-calibration workflow
                    logging.debug(f"[MAIN-AUTO-CAL] About to call handle_overlay_click for overlay {overlay_to_sample.key_id}")
                    success = self.auto_calibration_workflow.handle_overlay_click(overlay_to_sample, roi_bgr)
                    logging.debug(f"[MAIN-AUTO-CAL] handle_overlay_click returned: {success}")

                    if success:
                        logging.info(f"[MAIN-AUTO-CAL] Auto-calibration successful - showing success message")
                        QMessageBox.information(self.app, "Auto-Calibration Complete",
                                               f"Auto-calibration completed successfully for {overlay_to_sample.key_id}!")
                        logging.debug(f"[MAIN-AUTO-CAL] Triggering control panel update after successful calibration")
                        # Also explicitly trigger control panel update
                        self.control_panel.update_controls_from_state()
                        logging.debug(f"[MAIN-AUTO-CAL] Control panel update completed")
                    else:
                        logging.warning(f"[MAIN-AUTO-CAL] Auto-calibration failed - showing error message")
                        QMessageBox.warning(self.app, "Auto-Calibration Failed",
                                           "Auto-calibration failed. Please check the logs for details.")
                else:
                    QMessageBox.warning(self.app, "ROI Error", "Could not extract ROI data from the selected overlay.")
            else:
                QMessageBox.warning(self.app, "Calibration Error", f"Could not find overlay for key {selected_key_id} or auto-calibration workflow not available.")
                self.app_state.calibration.calibration_mode = None

        # Potentially update UI elements that depend on selected overlay
        self.control_panel.update_selected_overlay_display()
