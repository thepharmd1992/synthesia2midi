"""Shadow ROI and shadow calibration controller."""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.app_config import OverlayConfig

translate = QCoreApplication.translate


class ShadowCalibrationController:
    """Focused calibration controller extracted from the main window."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    def select_shadow_roi(self):
        """Handle shadow ROI selection request from control panel."""
        logging.info("Shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_roi_selection_mode()
                QMessageBox.information(
                    self.app,
                    translate("ShadowCalibrationController", "Shadow ROI Selection"),
                    translate(
                        "ShadowCalibrationController",
                        "Click and drag on the video to select the shadow detection region.\nShadow zones will be created for each key overlay.\nRight-click to cancel.",
                    ),
                )
            else:
                QMessageBox.warning(
                    self.app,
                    translate("ShadowCalibrationController", "Canvas Error"),
                    translate("ShadowCalibrationController", "Canvas interaction system not available."),
                )
        else:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Canvas"),
                translate("ShadowCalibrationController", "No video canvas available for ROI selection."),
            )

    def select_shadow_white_roi(self):
        """Handle white key shadow ROI selection request from control panel."""
        logging.info("White key shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_white_roi_selection_mode()
                QMessageBox.information(
                    self.app,
                    translate("ShadowCalibrationController", "White Key Shadow ROI Selection"),
                    translate(
                        "ShadowCalibrationController",
                        "Click and drag on the video to select the white key shadow detection region.\nThis will define the vertical region where white key shadows are detected.\nRight-click to cancel.",
                    ),
                )
            else:
                QMessageBox.warning(
                    self.app,
                    translate("ShadowCalibrationController", "Canvas Error"),
                    translate("ShadowCalibrationController", "Canvas interaction system not available."),
                )
        else:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Canvas"),
                translate("ShadowCalibrationController", "No video canvas available for ROI selection."),
            )

    def select_shadow_black_roi(self):
        """Handle black key shadow ROI selection request from control panel."""
        logging.info("Black key shadow ROI selection requested - entering ROI selection mode")
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            # Enter ROI selection mode on the canvas
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.enter_shadow_black_roi_selection_mode()
                QMessageBox.information(
                    self.app,
                    translate("ShadowCalibrationController", "Black Key Shadow ROI Selection"),
                    translate(
                        "ShadowCalibrationController",
                        "Click and drag on the video to select the black key shadow detection region.\nThis will define the vertical region where black key shadows are detected.\nRight-click to cancel.",
                    ),
                )
            else:
                QMessageBox.warning(
                    self.app,
                    translate("ShadowCalibrationController", "Canvas Error"),
                    translate("ShadowCalibrationController", "Canvas interaction system not available."),
                )
        else:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Canvas"),
                translate("ShadowCalibrationController", "No video canvas available for ROI selection."),
            )

    def set_shadow_detection_enabled(self, enabled: bool):
        """Handle shadow detection enable/disable toggle."""
        logging.info(f"Shadow detection {'enabled' if enabled else 'disabled'}")

    def set_shadow_detection_sensitivity(self, value: float):
        """Handle shadow detection sensitivity change."""
        logging.info(f"Shadow detection sensitivity changed to {value:.2f}")

    def set_shadow_darkness_threshold(self, value: float):
        """Handle shadow darkness threshold change."""
        logging.info(f"Shadow darkness threshold changed to {value:.2f}")

    def request_shadow_calibration(self, key_type: str, calibration_type: str):
        """Handle shadow calibration request for specific key type."""
        logging.info(f"Shadow calibration requested: key_type={key_type}, calibration_type={calibration_type}")

        if not hasattr(self, 'keyboard_canvas') or not self.keyboard_canvas:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Canvas"),
                translate("ShadowCalibrationController", "No video canvas available for calibration."),
            )
            return

        if self.keyboard_canvas.current_frame_rgb is None:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Frame"),
                translate(
                    "ShadowCalibrationController",
                    "No video frame loaded. Please open a video and navigate to a frame.",
                ),
            )
            return

        # Map calibration type and key type to calibration mode
        calibration_mode = f"shadow_{key_type.lower()}_{calibration_type}"

        # Set calibration mode
        self.app_state.calibration.calibration_mode = calibration_mode

        # Show instruction dialog
        key_type_display = {
            "LW": "Left White",
            "LB": "Left Black",
            "RW": "Right White",
            "RB": "Right Black"
        }.get(key_type, key_type)

        calibration_display = {
            "unpressed": "Unpressed (lit but not pressed)",
            "pressed": "Pressed (lit and fully pressed)"
        }.get(calibration_type, calibration_type)

        instruction_msg = translate(
            "ShadowCalibrationController",
            "Now click on a {key_type_display} key that shows {calibration_display} condition.\n\nFor {calibration_display}:\n",
        ).format(key_type_display=key_type_display, calibration_display=calibration_display)

        if calibration_type == "unpressed":
            instruction_msg += translate(
                "ShadowCalibrationController",
                "- Key should be lit (colored bars visible)\n- Key should NOT be pressed down\n- No shadow should be visible underneath",
            )
        else:  # pressed
            instruction_msg += translate(
                "ShadowCalibrationController",
                "- Key should be lit (colored bars visible)\n- Key should be FULLY pressed down\n- Dark shadow should be visible underneath",
            )

        QMessageBox.information(
            self.app,
            translate("ShadowCalibrationController", "Shadow Calibration - {key_type_display}").format(
                key_type_display=key_type_display
            ),
            instruction_msg,
        )
        logging.info(f"Set calibration mode to {calibration_mode}, waiting for user to click on key")

    def capture_shadow_overlay_calibration(self, overlay, calibration_mode: str):
        """Capture shadow calibration from selected overlay."""
        logging.info(f"Capturing {calibration_mode} calibration from overlay {overlay.key_id}")

        # Parse calibration mode: shadow_{key_type}_{unpressed/pressed}
        parts = calibration_mode.split('_')
        if len(parts) != 3 or parts[0] != 'shadow':
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "Invalid Mode"),
                translate(
                    "ShadowCalibrationController", "Invalid shadow calibration mode: {calibration_mode}"
                ).format(calibration_mode=calibration_mode),
            )
            self.app_state.calibration.calibration_mode = None
            return

        key_type = parts[1].upper()  # lw -> LW
        calibration_type = parts[2]  # unpressed or pressed

        logging.info(f"Shadow calibration for {key_type} type using key {overlay.key_id} ({getattr(overlay, 'note_name_in_octave', 'Unknown')})")

        # Note: LW/LB/RW/RB types are organizational helpers only, not technical requirements
        # Any key can be used to calibrate any type - the types are just visual guides for the user

        # Get the current frame
        current_frame = self.keyboard_canvas.current_frame_rgb
        if current_frame is None:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Frame"),
                translate("ShadowCalibrationController", "No frame available for calibration."),
            )
            self.app_state.calibration.calibration_mode = None
            return

        # Get shadow overlay for this key if it exists
        logging.info(f"=== SHADOW OVERLAY LOOKUP DEBUG ===")
        logging.info(f"Looking for shadow overlay for key {overlay.key_id}")
        logging.info(f"Total overlays in app_state: {len(self.app_state.overlays)}")

        # Log all overlays for debugging
        for i, ov in enumerate(self.app_state.overlays):
            overlay_type = getattr(ov, 'overlay_type', 'None')
            logging.info(f"  Overlay {i}: key_id={ov.key_id}, type={overlay_type}")

        shadow_overlay = None
        for ov in self.app_state.overlays:
            if hasattr(ov, 'overlay_type') and ov.overlay_type == 'shadow' and ov.key_id == overlay.key_id:
                shadow_overlay = ov
                logging.info(f"Found matching shadow overlay for key {overlay.key_id}")
                break

        if not shadow_overlay:
            logging.warning(f"No shadow overlay found for key {overlay.key_id}")
            logging.info(f"Shadow overlays in system:")
            shadow_overlays = [ov for ov in self.app_state.overlays if hasattr(ov, 'overlay_type') and ov.overlay_type == 'shadow']
            for i, sov in enumerate(shadow_overlays):
                logging.info(f"  Shadow overlay {i}: key_id={sov.key_id}")

            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "No Shadow Overlay"),
                translate(
                    "ShadowCalibrationController",
                    "No shadow overlay found for key {key_id}. Please create shadow overlays before calibration.",
                ).format(key_id=overlay.key_id),
            )
            self.app_state.calibration.calibration_mode = None
            return

        # Extract shadow ROI
        shadow_roi = self.extract_roi(current_frame, shadow_overlay)
        if shadow_roi is None or shadow_roi.size == 0:
            QMessageBox.warning(
                self.app,
                translate("ShadowCalibrationController", "ROI Error"),
                translate("ShadowCalibrationController", "Could not extract shadow region."),
            )
            self.app_state.calibration.calibration_mode = None
            return

        # Calculate darkness ratio for the shadow region
        gray = cv2.cvtColor(shadow_roi, cv2.COLOR_RGB2GRAY)
        black_pixel_threshold = 30  # Pixels below this are considered "black"
        black_pixels = np.sum(gray < black_pixel_threshold)
        total_pixels = gray.size
        darkness_ratio = black_pixels / total_pixels if total_pixels > 0 else 0.0

        # Store calibration data based on type
        calibration_data = {
            'darkness_ratio': darkness_ratio,
            'mean_brightness': np.mean(gray),
            'std_brightness': np.std(gray),
            'black_pixel_count': int(black_pixels),
            'total_pixels': int(total_pixels),
            'frame_index': self.app_state.video.current_frame_index
        }

        # Store in appropriate app_state field
        if calibration_type == 'unpressed':
            setattr(self.app_state.detection, f'shadow_calibration_{key_type.lower()}_unpressed', calibration_data)
            state_desc = "unpressed (no shadow)"
        else:  # pressed
            setattr(self.app_state.detection, f'shadow_calibration_{key_type.lower()}_pressed', calibration_data)
            state_desc = "pressed (with shadow)"

        # Update UI
        self.control_panel.update_shadow_calibration_display()

        # Auto-save calibration
        if hasattr(self, 'video_loading_workflow') and self.video_loading_workflow:
            success = self.video_loading_workflow.save_current_config()
            save_msg = (
                translate("ShadowCalibrationController", "\nCalibration data automatically saved.")
                if success
                else translate("ShadowCalibrationController", "\nWarning: Auto-save failed.")
            )
        else:
            save_msg = translate("ShadowCalibrationController", "\nWarning: Auto-save not available.")

        # Show success message
        QMessageBox.information(
            self.app,
            translate("ShadowCalibrationController", "Shadow Calibration - {key_type}").format(
                key_type=key_type
            ),
            translate(
                "ShadowCalibrationController",
                "Shadow calibration captured for {key_type} key in {state_desc} state.\n\nDarkness ratio: {darkness_ratio}\nMean brightness: {mean_brightness}\nBlack pixels: {black_pixels}/{total_pixels}{save_msg}",
            ).format(
                key_type=key_type,
                state_desc=state_desc,
                darkness_ratio=f"{darkness_ratio:.1%}",
                mean_brightness=f"{calibration_data['mean_brightness']:.1f}",
                black_pixels=black_pixels,
                total_pixels=total_pixels,
                save_msg=save_msg,
            ),
        )

        logging.info(f"Shadow calibration completed for {key_type} {calibration_type}: darkness_ratio={darkness_ratio:.3f}")

        # Reset calibration mode
        self.app_state.calibration.calibration_mode = None
        self.app_state.unsaved_changes = False  # Reset since we auto-saved

    def extract_roi(self, frame: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        """Extract region of interest from frame based on overlay coordinates."""
        if frame is None or overlay is None:
            return None

        x, y = int(overlay.x), int(overlay.y)
        w, h = int(overlay.width), int(overlay.height)

        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x1 >= x2 or y1 >= y2:
            return None

        return frame[y1:y2, x1:x2]

    # Backward-compatible private aliases for older callers/tests. New wiring
    # should use the public controller methods above.
    _handle_shadow_roi_selection_request = select_shadow_roi
    _handle_shadow_white_roi_selection_request = select_shadow_white_roi
    _handle_shadow_black_roi_selection_request = select_shadow_black_roi
    _handle_shadow_detection_toggle = set_shadow_detection_enabled
    _handle_shadow_detection_sensitivity_change = set_shadow_detection_sensitivity
    _handle_shadow_darkness_threshold_change = set_shadow_darkness_threshold
    _handle_shadow_calibration_request = request_shadow_calibration
    _capture_shadow_overlay_calibration = capture_shadow_overlay_calibration
    _extract_roi = extract_roi
