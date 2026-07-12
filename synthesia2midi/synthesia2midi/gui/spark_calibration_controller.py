"""Spark ROI and spark calibration controller."""
from __future__ import annotations

import logging

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.core.color_families import exemplar_display_parts

translate = QCoreApplication.translate


def _exemplar_display_label(slot: str) -> str:
    family_number, morphology = exemplar_display_parts(slot)
    family_label = translate(
        "SparkCalibrationController", "Color {number}"
    ).format(number=family_number)
    morphology_label = (
        translate("SparkCalibrationController", "Natural")
        if morphology == "Natural"
        else translate("SparkCalibrationController", "Sharp / Flat")
    )
    return f"{family_label} {morphology_label}"


class SparkCalibrationController:
    """Focused calibration controller extracted from the main window."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    def select_spark_roi(self):
        """Handle spark ROI selection request from control panel."""
        logging.info("Spark ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_spark_roi_selection_mode()
                QMessageBox.information(
                    self.app,
                    translate("SparkCalibrationController", "Spark ROI Selection"),
                    translate(
                        "SparkCalibrationController",
                        "Click and drag on the video to select the spark detection region.\nRight-click to cancel.",
                    ),
                )
            else:
                QMessageBox.warning(
                    self.app,
                    translate("SparkCalibrationController", "Canvas Error"),
                    translate("SparkCalibrationController", "Canvas interaction system not available."),
                )
        else:
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "No Canvas"),
                translate("SparkCalibrationController", "No video canvas available for ROI selection."),
            )

    def set_spark_roi_visible(self, visible: bool):
        """Handle spark ROI visibility toggle from control panel."""
        logging.info(f"Spark ROI visibility toggled to: {visible}")
        self.app_state.detection.spark_roi_visible = visible
        # Update canvas to show/hide the ROI
        if hasattr(self, 'display_manager') and self.display_manager:
            self.display_manager.refresh_canvas_overlays()

    def update_spark_roi_from_canvas(self, top_y: int, bottom_y: int):
        """Handle spark ROI coordinates updated from canvas selection."""
        logging.info(f"Spark ROI updated from canvas: top={top_y}, bottom={bottom_y}")

        self.app_state.detection.spark_roi_top = top_y
        self.app_state.detection.spark_roi_bottom = bottom_y
        self.app_state.detection.spark_roi_visible = True
        self.app_state.unsaved_changes = True

        try:
            from synthesia2midi.detection.spark_mapper import get_spark_mapper
            get_spark_mapper().invalidate_cache()
        except ImportError:
            pass

        if hasattr(self, 'display_manager') and self.display_manager:
            self.display_manager.refresh_canvas_overlays()

        # Update control panel to reflect new values
        if hasattr(self, 'control_panel') and self.control_panel:
            self.control_panel.update_controls_from_state()

        # Show confirmation message
        QMessageBox.information(
            self.app,
            translate("SparkCalibrationController", "Spark ROI Set"),
            translate(
                "SparkCalibrationController",
                "Spark detection region set:\nTop: {top_y} pixels\nBottom: {bottom_y} pixels\nHeight: {height} pixels",
            ).format(top_y=top_y, bottom_y=bottom_y, height=bottom_y - top_y),
        )

    def request_spark_calibration(self, step_type: str):
        """Handle spark calibration request from control panel."""
        logging.info(f"Spark calibration requested: {step_type}")

        if not hasattr(self, 'keyboard_canvas') or not self.keyboard_canvas:
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "No Canvas"),
                translate("SparkCalibrationController", "No video canvas available for calibration."),
            )
            return

        if self.keyboard_canvas.current_frame_rgb is None:
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "No Frame"),
                translate(
                    "SparkCalibrationController",
                    "No video frame loaded. Please open a video and navigate to a frame.",
                ),
            )
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
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "Invalid Step"),
                translate("SparkCalibrationController", "Unknown calibration step: {step_type}").format(
                    step_type=step_type
                ),
            )
            return

        calibration_mode = calibration_mode_map[step_type]

        # Handle background calibration differently (immediate capture)
        if step_type == "background":
            self.capture_spark_background_calibration()
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

        instruction_msg = translate(
            "SparkCalibrationController",
            "Now click on a {hand_type}key that shows {condition} condition.\n\nFor {display_name}:\n",
        ).format(hand_type=hand_type, condition=display_name.lower(), display_name=display_name)

        if "bar_only" in step_type:
            instruction_msg += translate(
                "SparkCalibrationController",
                "- Key should show colored bars WITHOUT any sparks\n- Navigate to a frame where this condition is visible\n- Click on the key showing this exact condition",
            )
        elif "dimmest_sparks" in step_type:
            instruction_msg += translate(
                "SparkCalibrationController",
                "- Key should show colored bars WITH barely visible sparks\n- Navigate to a frame where sparks are just starting to appear\n- Click on the key showing this exact condition",
            )
        else:  # brightest_sparks
            instruction_msg += translate(
                "SparkCalibrationController",
                "- Key should show colored bars WITH very bright/intense sparks\n- Navigate to a frame where sparks are at their brightest\n- Click on the key showing this exact condition",
            )

        QMessageBox.information(
            self.app,
            translate("SparkCalibrationController", "Spark {display_name} Calibration").format(
                display_name=display_name
            ),
            instruction_msg,
        )
        logging.info(f"Set calibration mode to {calibration_mode}, waiting for user to click on key")

    def start_auto_spark_calibration(self, key_type: str):
        """Handle auto-spark calibration request from control panel."""
        logging.info(f"Auto-spark calibration requested for key type: {key_type}")

        if not hasattr(self, 'auto_calibration_workflow') or not self.auto_calibration_workflow:
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "Workflow Error"),
                translate("SparkCalibrationController", "Auto-calibration workflow not available."),
            )
            return

        # Start the auto-calibration process
        success = self.auto_calibration_workflow.start_auto_calibration(key_type)

        if success:
            exemplar_label = _exemplar_display_label(key_type)
            instruction_msg = translate(
                "SparkCalibrationController",
                "Auto-calibration for {label} started.\n\n"
                "1. Navigate to a frame where a {label} key first turns on.\n"
                "2. Click that key overlay.\n"
                "3. The application will capture the bar-only frame, dimmest sparks, "
                "and brightest sparks, then save the calibration.",
            ).format(label=exemplar_label)

            QMessageBox.information(
                self.app,
                translate(
                    "SparkCalibrationController", "Auto-Calibrate {label}"
                ).format(label=exemplar_label),
                instruction_msg,
            )
        else:
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "Calibration Error"),
                translate(
                    "SparkCalibrationController",
                    "Failed to start auto-calibration. Please check video and overlays are loaded.",
                ),
            )

    def set_spark_detection_enabled(self, enabled: bool):
        """Handle spark detection enable/disable toggle."""
        logging.info(f"Spark detection {'enabled' if enabled else 'disabled'}")

        # Update the app state
        self.app_state.detection.spark_detection_enabled = enabled
        self.app_state.unsaved_changes = True

    def set_spark_detection_sensitivity(self, value: float):
        """Handle spark detection sensitivity change."""
        logging.info(f"Spark detection sensitivity changed to {value:.2f}")

        # Update the app state
        self.app_state.detection.spark_detection_sensitivity = value
        self.app_state.unsaved_changes = True

    def capture_spark_background_calibration(self):
        """Capture background calibration immediately (no user interaction needed)."""
        logging.info("Capturing spark background calibration")

        # Import calibration classes
        from synthesia2midi.detection.spark_calibration import SparkCalibrationManager, CalibrationStep

        # Create calibration manager
        calibration_manager = SparkCalibrationManager(self.app_state)

        # Start background calibration step
        if not calibration_manager.start_calibration_step(CalibrationStep.BACKGROUND):
            QMessageBox.warning(
                self.app,
                translate("SparkCalibrationController", "Calibration Failed"),
                translate(
                    "SparkCalibrationController",
                    "Could not start background calibration.\n\nRequirements:\n- Spark ROI must be set (top < bottom)\n- Key overlays must be configured",
                ),
            )
            return

        # Capture current frame
        current_frame = self.keyboard_canvas.current_frame_rgb
        frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)

        if calibration_manager.capture_calibration_frame(current_frame, frame_index, "spark_calibration_background"):
            # Update UI display
            self.control_panel.update_spark_calibration_display()

            # Show success message
            QMessageBox.information(
                self.app,
                translate("SparkCalibrationController", "Background Calibration Complete"),
                translate("SparkCalibrationController", "Background calibration captured successfully!"),
            )
            logging.info("Spark background calibration completed successfully")
        else:
            QMessageBox.critical(
                self.app,
                translate("SparkCalibrationController", "Calibration Failed"),
                translate(
                    "SparkCalibrationController",
                    "Failed to capture background calibration data.\n\nPlease check that spark ROI is properly set.",
                ),
            )
            logging.error("Spark background calibration failed")

    def capture_spark_overlay_calibration(self, overlay, calibration_mode: str):
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
                QMessageBox.warning(
                    self.app,
                    translate("SparkCalibrationController", "Calibration Error"),
                    translate(
                        "SparkCalibrationController",
                        "No spark zone found for key {key_id}. Please ensure spark ROI is properly set.",
                    ).format(key_id=overlay.key_id),
                )
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
                QMessageBox.warning(
                    self.app,
                    translate("SparkCalibrationController", "Invalid Mode"),
                    translate("SparkCalibrationController", "Unknown calibration mode: {calibration_mode}").format(
                        calibration_mode=calibration_mode
                    ),
                )
                self.app_state.calibration.calibration_mode = None
                return

            calibration_step, field_name = mode_map[calibration_mode]

            # Start calibration step
            if not calibration_manager.start_calibration_step(calibration_step):
                QMessageBox.warning(
                    self.app,
                    translate("SparkCalibrationController", "Calibration Failed"),
                    translate(
                        "SparkCalibrationController",
                        "Could not start calibration step. Please check spark ROI configuration.",
                    ),
                )
                self.app_state.calibration.calibration_mode = None
                return

            # Capture single zone calibration
            current_frame = self.keyboard_canvas.current_frame_rgb
            frame_index = getattr(self.keyboard_canvas, 'current_frame_index', 0)

            # Extract calibration sample from the target zone only
            zone_sample = calibration_manager._extract_zone_sample(current_frame, target_zone)
            if not zone_sample:
                QMessageBox.critical(
                    self.app,
                    translate("SparkCalibrationController", "Calibration Failed"),
                    translate(
                        "SparkCalibrationController", "Could not extract calibration data from key {key_id}."
                    ).format(key_id=overlay.key_id),
                )
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

            QMessageBox.information(
                self.app,
                translate("SparkCalibrationController", "{step_name} Calibration Complete").format(
                    step_name=step_name
                ),
                translate(
                    "SparkCalibrationController",
                    "{step_name} calibration captured successfully from key {key_id}!\n\nQuality: {quality}\nBrightness: {brightness}",
                ).format(
                    step_name=step_name,
                    key_id=overlay.key_id,
                    quality=f"{calib_data.confidence_score:.1%}",
                    brightness=f"{calib_data.mean_brightness:.3f}",
                ),
            )

            logging.info(f"Spark {calibration_mode} calibration completed successfully from key {overlay.key_id}")

        except Exception as e:
            logging.error(f"Error during spark calibration: {e}")
            QMessageBox.critical(
                self.app,
                translate("SparkCalibrationController", "Calibration Error"),
                translate(
                    "SparkCalibrationController", "An error occurred during calibration: {error}"
                ).format(error=str(e)),
            )
        finally:
            # Always reset calibration mode
            self.app_state.calibration.calibration_mode = None

    def get_calibration_instructions(self, step_type: str) -> str:
        """Get user instructions for each calibration step."""
        instructions = {
            "background": translate(
                "SparkCalibrationController",
                "Navigate to a frame with no bars visible and no sparks.\nThe spark ROI should show only background content.",
            ),
            "bar_only": translate(
                "SparkCalibrationController",
                "Navigate to a frame with colored bars visible but NO sparks.\nBars should be clearly visible in the spark ROI without any bright flashes.",
            ),
            "dimmest_sparks": translate(
                "SparkCalibrationController",
                "Navigate to a frame with the DIMMEST visible sparks.\nSparks should be just barely noticeable as bright flashes in the ROI.",
            ),
        }
        return instructions.get(step_type, translate("SparkCalibrationController", "Unknown calibration step"))

    # Backward-compatible private aliases for older callers/tests. New wiring
    # should use the public controller methods above.
    _handle_spark_roi_selection_request = select_spark_roi
    _handle_spark_roi_visibility_toggle = set_spark_roi_visible
    _handle_spark_roi_updated = update_spark_roi_from_canvas
    _handle_spark_calibration_request = request_spark_calibration
    _handle_auto_spark_calibration_request = start_auto_spark_calibration
    _handle_spark_detection_toggle = set_spark_detection_enabled
    _handle_spark_detection_sensitivity_change = set_spark_detection_sensitivity
    _capture_spark_background_calibration = capture_spark_background_calibration
    _capture_spark_overlay_calibration = capture_spark_overlay_calibration
    _get_calibration_instructions = get_calibration_instructions
