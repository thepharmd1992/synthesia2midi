"""
Calibration workflow.

Runs the calibration wizard and manages calibration-related actions.
"""
import logging
import os
from typing import Optional, Dict, Tuple, List

import numpy as np
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QInputDialog
from PySide6.QtCore import Qt

from synthesia2midi.video_loader import VideoSession
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.wizard import CalibrationWizard
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.detection.roi_utils import get_hist_feature
from synthesia2midi.gui.controls_qt import KEY_TYPES
from synthesia2midi.config_manager import ConfigManager
from synthesia2midi.app_config import (
    OverlayConfig, 
    DEFAULT_WHITE_KEY_STYLE, 
    DEFAULT_BLACK_KEY_STYLE,
    NOTE_NAMES_SHARP
)


class CalibrationWorkflow:
    """
    Handles calibration wizard process and overlay management.
    """
    
    def __init__(self, app_state: AppState, video_session: VideoSession, parent_widget=None):
        self.app_state = app_state
        self.video_session = video_session
        self.parent_widget = parent_widget
        self.logger = logging.getLogger(f"{__name__}.CalibrationWorkflow")
        
        # Detector instance (initialized on demand)
        self.detector = None
    
    def run_calibration_wizard(self) -> CalibrationWizard:
        """
        Run the calibration wizard to setup overlays.
        
        Returns:
            CalibrationWizard instance for signal connections
        """
        try:
            # Validate prerequisites
            if not self._validate_prerequisites():
                return None
            
            self.logger.info("Starting manual calibration wizard invocation.")
            
            # Note: We don't clear overlays here anymore because the wizard
            # will handle preserving unlit calibration data before clearing
            self.app_state.unsaved_changes = True
            
            # Create and return the wizard (don't exec it yet)
            wizard = CalibrationWizard(self.parent_widget, self.app_state)
            return wizard
                
        except Exception as e:
            self.logger.error(f"Calibration workflow failed: {e}")
            self._show_error("Calibration Error", f"Calibration failed: {e}")
            return None
    
    def handle_wizard_completed(self, wizard_success: bool) -> bool:
        """
        Handle wizard completion.
        
        Returns:
            True if calibration completed successfully, False otherwise
        """
        self.logger.info(f"Wizard completed. Success: {wizard_success}")
        
        if wizard_success and self.app_state.overlays:
            self.logger.info("Wizard submitted successfully and overlays generated.")
            self._apply_overlay_styles()
            self._update_ui_after_calibration()
            return True
        else:
            self.logger.warning("Calibration wizard failed or was cancelled")
            return False
    
    def _validate_prerequisites(self) -> bool:
        """Validate that calibration can proceed."""
        if not self.video_session:
            self._show_error("Wizard Error", "Please open a video file first.")
            return False
        
        return True
    
    def _apply_overlay_styles(self):
        """Apply styling to overlays after calibration."""
        # This would implement overlay styling logic
        # For now, just log that styles would be applied
        self.logger.info(f"Applied styles to {len(self.app_state.overlays)} overlays")
    
    def _update_ui_after_calibration(self):
        """Update UI state after successful calibration."""
        # Reset selection state
        self.app_state.ui.selected_overlay_id = None
        
        # Update calibration state
        self.app_state.calibration.calibration_mode = None
        self.app_state.calibration.current_calibration_key_type = None
        
        self.logger.info("UI state updated after calibration")
    
    def reset_calibration(self):
        """Reset calibration state and clear overlays."""
        self.app_state.overlays.clear()
        self.app_state.ui.selected_overlay_id = None
        self.app_state.calibration.calibration_mode = None
        self.app_state.calibration.current_calibration_key_type = None
        self.app_state.unsaved_changes = True
        
        self.logger.info("Calibration state reset")
    
    def validate_overlays(self) -> bool:
        """Validate that overlays are properly configured."""
        if not self.app_state.overlays:
            return False
        
        for overlay in self.app_state.overlays:
            if overlay.unlit_reference_color is None:
                self.logger.warning(f"Overlay {overlay.key_id} missing unlit reference color")
                return False
            
            if overlay.key_type is None:
                self.logger.warning(f"Overlay {overlay.key_id} missing key type")
                return False
        
        return True
    
    def get_calibration_summary(self) -> dict:
        """Get summary of current calibration state."""
        return {
            "total_overlays": len(self.app_state.overlays),
            "calibrated_overlays": len([o for o in self.app_state.overlays 
                                      if o.unlit_reference_color is not None]),
            "calibration_mode": self.app_state.calibration.calibration_mode,
            "current_key_type": self.app_state.calibration.current_calibration_key_type,
            "selected_overlay": self.app_state.ui.selected_overlay_id
        }
    



    def initialize_detector(self):
        """Initialize standard detector."""
        self.detector = DetectionFactory.create_detector('standard')
        self.logger.info("Initialized standard detection method")



    # Calibration actions
    
    def handle_color_pick(self, color_rgb: Tuple[int, int, int], coordinates: Tuple[int, int]):
        """Handle color picked from the canvas (Ctrl+click compatibility path)."""
        # This supports Ctrl+click color sampling. Lit exemplar calibration uses overlay selection.
        logging.info(f"handle_color_pick called (legacy Ctrl+click path) with color_rgb={color_rgb}, coordinates={coordinates}")
        
        # For lit exemplar calibration, users should use regular clicks, not Ctrl+clicks
        if self.app_state.calibration.calibration_mode == "lit_exemplar":
            logging.info("Lit exemplar calibration detected - users should use regular clicks instead of Ctrl+clicks")
            self._show_info("Calibration Tip", "For lit exemplar calibration, please use regular clicks on overlays instead of Ctrl+clicks.")
            return

        # Color-to-channel mapping is not used in this workflow.
        logging.info("Color-to-channel mapping is not used in this workflow.")

    def handle_calibrate_unlit_all_keys(self):
        """Calibrate unlit reference color for all overlays."""
        if not self.video_session:
            self._show_error("Calibration Error", "No video file is open.")
            return
        
        if not self.app_state.overlays:
            self._show_error("Calibration Error", "No key overlays are defined. Please run the calibration wizard first.")
            return
        
        keyboard_canvas = getattr(self.parent_widget, 'keyboard_canvas', None)
        if not keyboard_canvas:
            self._show_error("Calibration Error", "Canvas not available.")
            return
        
        overlays_calibrated = 0
        for overlay in self.app_state.overlays:
            roi_bgr = keyboard_canvas.get_roi_bgr(overlay)
            if roi_bgr is not None:
                # Log ROI info
                logging.info(f"[UNLIT-CALIB] Key {overlay.key_id}: Got ROI with shape {roi_bgr.shape}")
                
                # Capture histogram feature
                overlay.unlit_hist = get_hist_feature(roi_bgr)
                logging.info(f"[UNLIT-CALIB] Key {overlay.key_id}: Set unlit_hist with shape {overlay.unlit_hist.shape if overlay.unlit_hist is not None else None}")
                
                # Capture average color (convert BGR to RGB)
                average_bgr = roi_bgr.mean(axis=(0, 1)).astype(int)
                overlay.unlit_reference_color = (int(average_bgr[2]), int(average_bgr[1]), int(average_bgr[0]))  # Convert BGR to RGB
                logging.info(f"[UNLIT-CALIB] Key {overlay.key_id}: Set unlit_reference_color to {overlay.unlit_reference_color}")
                
                overlays_calibrated += 1
                logging.info(f"Captured unlit_hist and unlit_reference_color {overlay.unlit_reference_color} for overlay {overlay.key_id}")
            else:
                logging.warning(f"Could not get ROI BGR for overlay {overlay.key_id}")
        
        if overlays_calibrated > 0:
            self.app_state.unsaved_changes = True
            
            # Auto-save calibration changes
            self._auto_save_calibration_changes()
            
            self._show_info("Unlit Calibration Complete", 
                          f"Calibrated unlit reference colors for {overlays_calibrated} key(s).\nCalibration data automatically saved.")
            logging.info(f"Unlit calibration completed for {overlays_calibrated} overlays.")
            
            # Update control panel to reflect calibration status
            if hasattr(self.parent_widget, 'control_panel') and self.parent_widget.control_panel:
                self.parent_widget.control_panel.update_controls_from_state()
        else:
            self._show_error("Calibration Failed", "No overlays could be calibrated.")

    def handle_calibrate_lit_exemplar_key_start(self, key_type: str):
        """Start lit exemplar calibration mode for a specific key type."""
        self.logger.debug(f"[HUE-CALIBRATION] Starting lit exemplar calibration for key type: {key_type}")
        
        # Check if it's a valid key type (base types, mono types, or COLOR_N_W/B format)
        is_valid = (
            key_type in KEY_TYPES or 
            (key_type.startswith("COLOR_") and 
             (key_type.endswith("_W") or key_type.endswith("_B")))
        )
        
        if not is_valid:
            self.logger.debug(f"[MONO-DEBUG] Invalid key type detected: {key_type}")
            self.logger.debug(f"[MONO-DEBUG] KEY_TYPES = {KEY_TYPES}")
            self.logger.debug(f"[MONO-DEBUG] Expected mono types: ['W', 'B']")
            self._show_error("Calibration Error", f"Invalid key type: {key_type}")
            self.logger.error(f"[CALIBRATION] ERROR: Invalid key type: {key_type}. Valid types: {KEY_TYPES + ['W', 'B']} or COLOR_N_W/B")
            return
        
        self.logger.debug(f"[MONO-DEBUG] Calibration workflow received key type: {key_type}")
        self.logger.debug(f"[MONO-DEBUG] Valid key types check passed")
        self.logger.info(f"[CALIBRATION] Starting calibration for key type: {key_type}")
        
        # Log current state before setting
        self.logger.debug(f"[HUE-CALIBRATION] Current hand detection state before calibration:")
        self.logger.debug(f"[HUE-CALIBRATION]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
        self.logger.debug(f"[HUE-CALIBRATION]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
        self.logger.debug(f"[HUE-CALIBRATION]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
        
        self.app_state.calibration.calibration_mode = "lit_exemplar"
        self.app_state.calibration.current_calibration_key_type = key_type
        
        # Generate human-readable key type name
        if key_type.startswith("COLOR_"):
            parts = key_type.split("_")
            if len(parts) == 3:
                color_num = parts[1]
                key_color = "White" if parts[2] == "W" else "Black"
                key_type_display = f"Color {color_num} {key_color}"
            else:
                key_type_display = key_type
        else:
            # Standard key types
            key_type_map = {
                "LW": "Left Hand White",
                "LB": "Left Hand Black", 
                "RW": "Right Hand White",
                "RB": "Right Hand Black",
                "W": "White Keys",
                "B": "Black Keys"
            }
            key_type_display = key_type_map.get(key_type, key_type)
        
        self._show_info("Lit Exemplar Calibration", 
                       f"Click on a lit {key_type_display} key in the video frame to calibrate. "
                       f"The application will sample the color and histogram for {key_type_display}.")
        logging.info(f"Starting lit exemplar calibration for key type: {key_type}")
        self.logger.debug(f"[HUE-CALIBRATION] Calibration mode set to 'lit_exemplar' for {key_type}")

    def apply_template_styles_to_overlays(self):
        """Apply template styles from my_immortal.ini or defaults to current overlays."""
        template_ini_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "my_immortal.ini")
        template_overlays = None
        if os.path.exists(template_ini_path):
            logging.info(f"Attempting to apply style from template: {template_ini_path}")
            config_manager = ConfigManager(self.app_state)
            template_overlays = config_manager.parse_overlays_from_file(template_ini_path)
        
        if template_overlays:
            self.apply_template_style(self.app_state.overlays, template_overlays)
        else:
            logging.info("No template INI found/parsed or it was empty. Applying default hardcoded styles.")
            # TODO: Call a new method like self._apply_default_styles(self.app_state.overlays)
            # For now, this else block is a placeholder for when we add default styles.
            pass

    def apply_template_style(self, target_overlays: List[OverlayConfig], template_overlays: Optional[List[OverlayConfig]]):
        """Applies Y and Height from template_overlays to target_overlays (wizard-generated), falling back to defaults."""
        if not target_overlays:
            logging.info("apply_template_style: No target overlays to process.")
            return

        # Initialize styles with defaults
        current_white_y = DEFAULT_WHITE_KEY_STYLE["y"]
        current_white_height = DEFAULT_WHITE_KEY_STYLE["height"]
        current_black_y = DEFAULT_BLACK_KEY_STYLE["y"]
        current_black_height = DEFAULT_BLACK_KEY_STYLE["height"]
        
        source_of_style = "defaults"
        
        if template_overlays:
            # Extract styles from template overlays
            white_key_note_names = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
            
            # Find first white and black overlay in template to get Y and Height
            template_white = next((o for o in template_overlays if o.note_name_in_octave in white_key_note_names), None)
            template_black = next((o for o in template_overlays if o.note_name_in_octave not in white_key_note_names), None)
            
            if template_white:
                current_white_y = template_white.y
                current_white_height = template_white.height
                source_of_style = "template"
                
            if template_black:
                current_black_y = template_black.y
                current_black_height = template_black.height
                source_of_style = "template"
        
        # Apply styles to target overlays
        white_key_note_names = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
        overlays_modified = 0
        
        for overlay in target_overlays:
            is_white_key = overlay.note_name_in_octave in white_key_note_names
            target_y = current_white_y if is_white_key else current_black_y
            target_height = current_white_height if is_white_key else current_black_height
            
            if overlay.y != target_y or overlay.height != target_height:
                overlay.y = target_y
                overlay.height = target_height
                overlays_modified += 1
                
        if overlays_modified > 0:
            self.app_state.unsaved_changes = True
            logging.info(f"Applied {source_of_style} Y/Height styles to {overlays_modified} overlays.")
        else:
            logging.info(f"No overlay Y/Height styles needed to be applied (source: {source_of_style}), or target/template styles already matched.")

    def handle_set_calibration_start(self, update_display_callback=None):
        """Set calibration start frame to current scrubber position."""
        if not self.video_session:
            self._show_error("Set Start Frame", "No video file is open.")
            return
        
        self.app_state.calibration.calib_start_frame = self.app_state.video.current_frame_index
        self.app_state.unsaved_changes = True
        
        # Update display via callback
        if update_display_callback:
            update_display_callback()
        
        logging.info(f"Calibration start frame set to: {self.app_state.calibration.calib_start_frame}")

    def handle_set_calibration_end(self, update_display_callback=None):
        """Set calibration end frame to current scrubber position."""
        if not self.video_session:
            self._show_error("Set End Frame", "No video file is open.")
            return
        
        self.app_state.calibration.calib_end_frame = self.app_state.video.current_frame_index
        self.app_state.unsaved_changes = True
        
        # Update display via callback
        if update_display_callback:
            update_display_callback()
        
        logging.info(f"Calibration end frame set to: {self.app_state.calibration.calib_end_frame}")

    def _show_info(self, title: str, message: str):
        """Show info message (if parent widget available)."""
        if self.parent_widget:
            QMessageBox.information(self.parent_widget, title, message)
        else:
            self.logger.info(f"{title}: {message}")

    def _show_error(self, title: str, message: str):
        """Show error message (if parent widget available)."""
        if self.parent_widget:
            QMessageBox.warning(self.parent_widget, title, message)
        else:
            self.logger.error(f"{title}: {message}")

    def _auto_save_calibration_changes(self):
        """Auto-save calibration changes to config file."""
        logging.info("_auto_save_calibration_changes called")
        
        # Debug: Check what exemplar histograms we have before saving
        for key_type, hist in self.app_state.detection.exemplar_lit_histograms.items():
            if hist is not None:
                logging.info(f"Before save: exemplar_lit_histograms[{key_type}] = {type(hist)} shape={hist.shape} sum={hist.sum()}")
            else:
                logging.info(f"Before save: exemplar_lit_histograms[{key_type}] = None")
        
        try:
            # Get the video_loading_workflow from parent widget to access save functionality
            video_loading_workflow = getattr(self.parent_widget, 'video_loading_workflow', None)
            if video_loading_workflow:
                logging.info("Found video_loading_workflow, calling save_current_config()")
                success = video_loading_workflow.save_current_config()
                if success:
                    logging.info("Calibration changes automatically saved to config file.")
                    self.app_state.unsaved_changes = False
                else:
                    logging.warning("Auto-save of calibration changes failed.")
            else:
                logging.warning("Cannot auto-save: video_loading_workflow not available.")
        except Exception as e:
            logging.error(f"Error during auto-save of calibration changes: {e}")
