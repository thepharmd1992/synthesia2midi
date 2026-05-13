"""Spark ROI and spark calibration controller."""
from __future__ import annotations

import logging

from PySide6.QtWidgets import QMessageBox


class SparkCalibrationController:
    """Focused calibration controller extracted from the main window."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    def _handle_spark_roi_selection_request(self):
        """Handle spark ROI selection request from control panel."""
        logging.info("Spark ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_spark_roi_selection_mode()
                QMessageBox.information(self.app, "Spark ROI Selection",
                                       "Click and drag on the video to select the spark detection region.\n"
                                       "Right-click to cancel.")
            else:
                QMessageBox.warning(self.app, "Canvas Error", "Canvas interaction system not available.")
        else:
            QMessageBox.warning(self.app, "No Canvas", "No video canvas available for ROI selection.")

    def _handle_spark_roi_visibility_toggle(self, visible: bool):
        """Handle spark ROI visibility toggle from control panel."""
        logging.info(f"Spark ROI visibility toggled to: {visible}")
        self.app_state.detection.spark_roi_visible = visible
        # Update canvas to show/hide the ROI
        if self.keyboard_canvas:
            self.keyboard_canvas.update()

    def _handle_spark_roi_updated(self, top_y: int, bottom_y: int):
        """Handle spark ROI coordinates updated from canvas selection."""
        logging.info(f"Spark ROI updated from canvas: top={top_y}, bottom={bottom_y}")

        # Update control panel to reflect new values
        if hasattr(self, 'control_panel') and self.control_panel:
            self.control_panel.update_controls_from_state()

        # Show confirmation message
        QMessageBox.information(self.app, "Spark ROI Set",
                               f"Spark detection region set:\n"
                               f"Top: {top_y} pixels\n"
                               f"Bottom: {bottom_y} pixels\n"
                               f"Height: {bottom_y - top_y} pixels")

    def _handle_spark_calibration_request(self, step_type: str):
        """Handle spark calibration request from control panel."""
        logging.info(f"Spark calibration requested: {step_type}")

        if not hasattr(self, 'keyboard_canvas') or not self.keyboard_canvas:
            QMessageBox.warning(self.app, "No Canvas", "No video canvas available for calibration.")
            return

        if self.keyboard_canvas.current_frame_rgb is None:
            QMessageBox.warning(self.app, "No Frame", "No video frame loaded. Please open a video and navigate to a frame.")
            return

        # Map step type to calibration modes
        calibration_mode_map = {
            "background": "spark_background",
            "bar_only": "spark_bar_only",
            "dimmest_sparks": "spark_dimmest_sparks",
            "lh_bar_only": "spark_lh_bar_only",
            "lh_dimmest_sparks": "spark_lh_dimmest_sparks",
            "lh_brightest_sparks": "spark_lh_brightest_sparks",
            "rh_bar_only": "spark_rh_bar_only",
            "rh_dimmest_sparks": "spark_rh_dimmest_sparks",
            "rh_brightest_sparks": "spark_rh_brightest_sparks"
        }

        if step_type not in calibration_mode_map:
            QMessageBox.warning(self.app, "Invalid Step", f"Unknown calibration step: {step_type}")
            return

        calibration_mode = calibration_mode_map[step_type]

        # Handle background calibration differently (immediate capture)
        if step_type == "background":
            self._capture_spark_background_calibration()
            return

        # For bar_only and dimmest_sparks, set mode and wait for user click
        self.app_state.calibration.calibration_mode = calibration_mode

        # Show instruction dialog
        step_display_names = {
            "bar_only": "Bar-Only",
            "dimmest_sparks": "Dimmest Sparks",
            "lh_bar_only": "Left Hand Bar-Only",
            "lh_dimmest_sparks": "Left Hand Dimmest Sparks",
            "lh_brightest_sparks": "Left Hand Brightest Sparks",
            "rh_bar_only": "Right Hand Bar-Only",
            "rh_dimmest_sparks": "Right Hand Dimmest Sparks",
            "rh_brightest_sparks": "Right Hand Brightest Sparks"
        }

        display_name = step_display_names.get(step_type, step_type)

        # Determine hand type
        hand_type = ""
        if "lh_" in step_type:
            hand_type = "LEFT HAND "
        elif "rh_" in step_type:
            hand_type = "RIGHT HAND "

        instruction_msg = (f"Now click on a {hand_type}key that shows {display_name.lower()} condition.\n\n"
                          f"For {display_name}:\n")

        if "bar_only" in step_type:
            instruction_msg += ("- Key should show colored bars WITHOUT any sparks\n"
                              "- Navigate to a frame where this condition is visible\n"
                              "- Click on the key showing this exact condition")
        elif "dimmest_sparks" in step_type:
            instruction_msg += ("- Key should show colored bars WITH barely visible sparks\n"
                              "- Navigate to a frame where sparks are just starting to appear\n"
                              "- Click on the key showing this exact condition")
        else:  # brightest_sparks
            instruction_msg += ("- Key should show colored bars WITH very bright/intense sparks\n"
                              "- Navigate to a frame where sparks are at their brightest\n"
                              "- Click on the key showing this exact condition")

        QMessageBox.information(self.app, f"Spark {display_name} Calibration", instruction_msg)
        logging.info(f"Set calibration mode to {calibration_mode}, waiting for user to click on key")

    def _handle_auto_spark_calibration_request(self, key_type: str):
        """Handle auto-spark calibration request from control panel."""
        logging.info(f"Auto-spark calibration requested for key type: {key_type}")

        if not hasattr(self, 'auto_calibration_workflow') or not self.auto_calibration_workflow:
            QMessageBox.warning(self.app, "Workflow Error", "Auto-calibration workflow not available.")
            return

        # Start the auto-calibration process
        success = self.auto_calibration_workflow.start_auto_calibration(key_type)

        if success:
            instruction_msg = (f"Auto-Calibration for {key_type} Started\n\n"
                              f"Instructions:\n"
                              f"1. Navigate to a frame where a {key_type} key FIRST turns ON\n"
                              f"2. Click on that key overlay\n"
                              f"3. The system will automatically:\n"
                              f"   - Detect if it's left/right hand based on color\n"
                              f"   - Capture bar-only (frame +0)\n"
                              f"   - Capture dimmest sparks (frame +2)\n"
                              f"   - Find brightest sparks (frames +3 to +22)\n"
                              f"   - Save calibration data\n\n"
                              f"Key Type Legend:\n"
                              f"LW = Left White, LB = Left Black\n"
                              f"RW = Right White, RB = Right Black")

            QMessageBox.information(self.app, f"Auto-Calibrate {key_type}", instruction_msg)
        else:
            QMessageBox.warning(self.app, "Calibration Error",
                               "Failed to start auto-calibration. Please check video and overlays are loaded.")

    def _handle_spark_detection_toggle(self, enabled: bool):
        """Handle spark detection enable/disable toggle."""
        logging.info(f"Spark detection {'enabled' if enabled else 'disabled'}")

        # Update the app state
        self.app_state.detection.spark_detection_enabled = enabled
        self.app_state.unsaved_changes = True

    def _handle_spark_detection_sensitivity_change(self, value: float):
        """Handle spark detection sensitivity change."""
        logging.info(f"Spark detection sensitivity changed to {value:.2f}")

        # Update the app state
        self.app_state.detection.spark_detection_sensitivity = value
        self.app_state.unsaved_changes = True

    def _capture_spark_background_calibration(self):
        """Capture background calibration immediately (no user interaction needed)."""
        logging.info("Capturing spark background calibration")

        # Import calibration classes
        from synthesia2midi.detection.spark_calibration import SparkCalibrationManager, CalibrationStep

        # Create calibration manager
        calibration_manager = SparkCalibrationManager(self.app_state)

        # Start background calibration step
        if not calibration_manager.start_calibration_step(CalibrationStep.BACKGROUND):
            QMessageBox.warning(self.app, "Calibration Failed",
                              "Could not start background calibration.\n\n"
                              "Requirements:\n"
                              "- Spark ROI must be set (top < bottom)\n"
                              "- Key overlays must be configured")
            return

        # Capture current frame
        current_frame = self.keyboard_canvas.current_frame_rgb
        frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)

        if calibration_manager.capture_calibration_frame(current_frame, frame_index, "spark_calibration_background"):
            # Update UI display
            self.control_panel.update_spark_calibration_display()

            # Show success message
            QMessageBox.information(self.app, "Background Calibration Complete",
                                   "Background calibration captured successfully!")
            logging.info("Spark background calibration completed successfully")
        else:
            QMessageBox.critical(self.app, "Calibration Failed",
                               "Failed to capture background calibration data.\n\n"
                               "Please check that spark ROI is properly set.")
            logging.error("Spark background calibration failed")

    def _capture_spark_overlay_calibration(self, overlay, calibration_mode: str):
        """Capture spark calibration from selected overlay's spark zone."""
        logging.info(f"Capturing {calibration_mode} calibration from overlay {overlay.key_id}")

        # Import calibration classes
        from synthesia2midi.detection.spark_calibration import SparkCalibrationManager, CalibrationStep
        from synthesia2midi.detection.spark_mapper import get_spark_zones

        try:
            # Get spark zones to find the zone for this overlay
            spark_zones = get_spark_zones(self.app_state)
            target_zone = next((zone for zone in spark_zones if zone.key_id == overlay.key_id), None)

            if not target_zone:
                QMessageBox.warning(self.app, "Calibration Error",
                                   f"No spark zone found for key {overlay.key_id}. "
                                   f"Please ensure spark ROI is properly set.")
                self.app_state.calibration.calibration_mode = None
                return

            # Create calibration manager
            calibration_manager = SparkCalibrationManager(self.app_state)

            # Map calibration mode to step and field
            mode_map = {
                "spark_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_bar_only"),
                "spark_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_dimmest_sparks"),
                "spark_lh_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_lh_bar_only"),
                "spark_lh_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_lh_dimmest_sparks"),
                "spark_lh_brightest_sparks": (CalibrationStep.BRIGHTEST_SPARKS, "spark_calibration_lh_brightest_sparks"),
                "spark_rh_bar_only": (CalibrationStep.BAR_ONLY, "spark_calibration_rh_bar_only"),
                "spark_rh_dimmest_sparks": (CalibrationStep.DIMMEST_SPARKS, "spark_calibration_rh_dimmest_sparks"),
                "spark_rh_brightest_sparks": (CalibrationStep.BRIGHTEST_SPARKS, "spark_calibration_rh_brightest_sparks")
            }

            if calibration_mode not in mode_map:
                QMessageBox.warning(self.app, "Invalid Mode", f"Unknown calibration mode: {calibration_mode}")
                self.app_state.calibration.calibration_mode = None
                return

            calibration_step, field_name = mode_map[calibration_mode]

            # Start calibration step
            if not calibration_manager.start_calibration_step(calibration_step):
                QMessageBox.warning(self.app, "Calibration Failed",
                                   "Could not start calibration step. Please check spark ROI configuration.")
                self.app_state.calibration.calibration_mode = None
                return

            # Capture single zone calibration
            current_frame = self.keyboard_canvas.current_frame_rgb
            frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)

            # Extract calibration sample from the target zone only
            zone_sample = calibration_manager._extract_zone_sample(current_frame, target_zone)
            if not zone_sample:
                QMessageBox.critical(self.app, "Calibration Failed",
                                   f"Could not extract calibration data from key {overlay.key_id}.")
                self.app_state.calibration.calibration_mode = None
                return

            # Create calibration data from single zone sample
            zone_samples = {target_zone.key_id: zone_sample}
            calib_data = calibration_manager._create_calibration_data(
                calibration_step, frame_index, zone_samples
            )

            # Store calibration data
            calibration_manager._store_calibration_data(calib_data, field_name)

            # Update UI display
            self.control_panel.update_spark_calibration_display()

            # Show success message
            step_names = {
                "spark_bar_only": "Bar-Only",
                "spark_dimmest_sparks": "Dimmest Sparks",
                "spark_lh_bar_only": "Left Hand Bar-Only",
                "spark_lh_dimmest_sparks": "Left Hand Dimmest Sparks",
                "spark_lh_brightest_sparks": "Left Hand Brightest Sparks",
                "spark_rh_bar_only": "Right Hand Bar-Only",
                "spark_rh_dimmest_sparks": "Right Hand Dimmest Sparks",
                "spark_rh_brightest_sparks": "Right Hand Brightest Sparks"
            }
            step_name = step_names.get(calibration_mode, calibration_mode)

            QMessageBox.information(self.app, f"{step_name} Calibration Complete",
                                   f"{step_name} calibration captured successfully from key {overlay.key_id}!\n\n"
                                   f"Quality: {calib_data.confidence_score:.1%}\n"
                                   f"Brightness: {calib_data.mean_brightness:.3f}")

            logging.info(f"Spark {calibration_mode} calibration completed successfully from key {overlay.key_id}")

        except Exception as e:
            logging.error(f"Error during spark calibration: {e}")
            QMessageBox.critical(self.app, "Calibration Error",
                               f"An error occurred during calibration: {str(e)}")
        finally:
            # Always reset calibration mode
            self.app_state.calibration.calibration_mode = None

    def _get_calibration_instructions(self, step_type: str) -> str:
        """Get user instructions for each calibration step."""
        instructions = {
            "background": "Navigate to a frame with no bars visible and no sparks.\nThe spark ROI should show only background content.",
            "bar_only": "Navigate to a frame with colored bars visible but NO sparks.\nBars should be clearly visible in the spark ROI without any bright flashes.",
            "dimmest_sparks": "Navigate to a frame with the DIMMEST visible sparks.\nSparks should be just barely noticeable as bright flashes in the ROI."
        }
        return instructions.get(step_type, "Unknown calibration step")
