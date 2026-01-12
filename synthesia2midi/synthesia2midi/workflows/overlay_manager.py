"""
Overlay management for Synthesia2MIDI application.

This module handles overlay manipulation operations such as alignment and key type changes.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from ..core.app_state import AppState
from ..app_config import OverlayConfig, NOTE_NAMES_SHARP
from ..gui.controls_qt import KEY_TYPES
from ..gui.ui_update_interface import UIUpdateInterface


class OverlayManager:
    """Manages overlay manipulation operations."""
    
    def __init__(self, app_state: AppState, ui_updater: Optional[UIUpdateInterface] = None):
        """
        Initialize OverlayManager.
        
        Args:
            app_state: The application state object
            ui_updater: UI update interface for clean dependency injection
        """
        self.app_state = app_state
        self.ui_updater = ui_updater
        self.logger = logging.getLogger(f"{__name__}.OverlayManager")
        self.parent_widget = None  # Will be set by main.py if needed
    
    def handle_selected_overlay_key_type_change(self, key_id: int, new_key_type: str):
        """Change the key type of a selected overlay."""
        if new_key_type not in KEY_TYPES:
            self.logger.error(f"Invalid key_type {new_key_type} for overlay {key_id}.")
            return
        
        overlay_to_update = next((o for o in self.app_state.overlays if o.key_id == key_id), None)
        if overlay_to_update:
            overlay_to_update.key_type = new_key_type
            self.app_state.unsaved_changes = True
            self.logger.info(f"Overlay {key_id} key_type changed to {new_key_type}.")
            
            # Refresh the display
            if self.ui_updater:
                self.ui_updater.update_selected_overlay_display()
        else:
            self.logger.warning(f"Could not find overlay with key_id {key_id} to update key_type.")

    def align_overlays_vertically(self, master_overlay: OverlayConfig, target_key_color_type: str):
        """Helper to align overlays of a given color type (W or B) to a master overlay's Y and Height."""
        if not master_overlay:
            return

        white_key_note_names = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
        modified_count = 0

        for overlay in self.app_state.overlays:
            if overlay.key_id == master_overlay.key_id:  # Don't align master to itself
                continue

            is_white_key = overlay.note_name_in_octave in white_key_note_names
            current_key_color_type = "W" if is_white_key else "B"

            if current_key_color_type == target_key_color_type:
                if overlay.y != master_overlay.y or overlay.height != master_overlay.height:
                    overlay.y = master_overlay.y
                    overlay.height = master_overlay.height
                    modified_count += 1
        
        if modified_count > 0:
            self.app_state.unsaved_changes = True
            
            # Redraw frame if canvas is available
            if self.ui_updater and self.app_state.video.current_frame_index is not None:
                self.ui_updater.refresh_canvas()
            
            # Show success message
            if self.ui_updater:
                self.ui_updater.show_message(
                    "Alignment Complete", 
                    f"{modified_count} '{target_key_color_type}' keys aligned vertically to overlay {master_overlay.key_id}."
                )
            
            self.logger.info(f"Aligned {modified_count} '{target_key_color_type}' keys vertically to overlay {master_overlay.key_id}.")
        else:
            if self.parent_widget:
                QMessageBox.information(
                    self.parent_widget, 
                    "Alignment Info", 
                    f"No other '{target_key_color_type}' keys needed vertical alignment."
                )

    def handle_align_white_keys_to_selected(self):
        """Align all white keys to the selected overlay's Y and Height."""
        if self.app_state.ui.selected_overlay_id is None:
            if self.parent_widget:
                QMessageBox.warning(self.parent_widget, "Alignment Error", "No overlay selected to act as master.")
            return
        
        master_overlay = next(
            (o for o in self.app_state.overlays if o.key_id == self.app_state.ui.selected_overlay_id), 
            None
        )
        if not master_overlay:
            if self.parent_widget:
                QMessageBox.critical(
                    self.parent_widget, 
                    "Alignment Error", 
                    f"Selected overlay ID {self.app_state.ui.selected_overlay_id} not found."
                )
            return
        
        self.align_overlays_vertically(master_overlay, "W")

    def handle_align_black_keys_to_selected(self):
        """Align all black keys to the selected overlay's Y and Height."""
        if self.app_state.ui.selected_overlay_id is None:
            if self.parent_widget:
                QMessageBox.warning(self.parent_widget, "Alignment Error", "No overlay selected to act as master.")
            return
        
        master_overlay = next(
            (o for o in self.app_state.overlays if o.key_id == self.app_state.ui.selected_overlay_id), 
            None
        )
        if not master_overlay:
            if self.parent_widget:
                QMessageBox.critical(
                    self.parent_widget, 
                    "Alignment Error", 
                    f"Selected overlay ID {self.app_state.ui.selected_overlay_id} not found."
                )
            return
        
        self.align_overlays_vertically(master_overlay, "B")

    def handle_overlay_selection(self, selected_key_id: Optional[int]):
        """
        Handle overlay selection with calibration mode interaction.
        
        This function manages overlay selection state and handles special
        calibration mode behavior for color sampling.
        """
        # Update selection state
        self.app_state.ui.selected_overlay_id = selected_key_id
        self.logger.debug(f"Overlay selected: key_id {selected_key_id}")

        # Handle calibration mode interaction
        if (self.app_state.calibration.calibration_mode == "lit_exemplar" and 
            self.app_state.calibration.current_calibration_key_type and 
            selected_key_id is not None):
            
            overlay_to_sample = next((o for o in self.app_state.overlays if o.key_id == selected_key_id), None)
            
            # Access keyboard canvas through parent widget
            keyboard_canvas = getattr(self.parent_widget, 'keyboard_canvas', None) if self.parent_widget else None
            
            if overlay_to_sample and keyboard_canvas:
                sampled_color = keyboard_canvas.get_average_color_for_overlay(
                    keyboard_canvas.current_frame_rgb, 
                    overlay_to_sample
                )
                
                if sampled_color:
                    key_type_to_cal = self.app_state.calibration.current_calibration_key_type
                    self.app_state.detection.exemplar_lit_colors[key_type_to_cal] = sampled_color
                    self.logger.info(f"Calibrated exemplar lit color for {key_type_to_cal} to {sampled_color} from overlay {selected_key_id}.")
                    self.logger.debug(f"[HUE-CALIBRATION] Calibrated exemplar lit color for {key_type_to_cal} to RGB{sampled_color}")
                    
                    # Extract hue for hand detection calibration (only if hand assignment is enabled)
                    if self.app_state.detection.hand_assignment_enabled:
                        roi_bgr = keyboard_canvas.get_roi_bgr(overlay_to_sample)
                        if roi_bgr is not None:
                            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                            avg_hue = np.mean(hsv[:, :, 0])
                            self.logger.debug(f"[HUE-CALIBRATION] Extracted hue value: {avg_hue:.1f} for {key_type_to_cal}")
                            
                            # Update hand detection calibration based on key type
                            if key_type_to_cal.startswith('L'):  # Left hand (LW or LB)
                                old_value = self.app_state.detection.left_hand_hue_mean
                                # Update left hand hue mean (running average if already calibrated)
                                if self.app_state.detection.left_hand_hue_mean > 0:
                                    self.app_state.detection.left_hand_hue_mean = (
                                        self.app_state.detection.left_hand_hue_mean + avg_hue) / 2
                                    self.logger.debug(f"[HUE-CALIBRATION] LEFT hand: Updated from {old_value:.1f} to {self.app_state.detection.left_hand_hue_mean:.1f} (average with {avg_hue:.1f})")
                                else:
                                    self.app_state.detection.left_hand_hue_mean = avg_hue
                                    self.logger.debug(f"[HUE-CALIBRATION] LEFT hand: Set initial value to {avg_hue:.1f}")
                                self.logger.info(f"Updated left hand hue mean to {self.app_state.detection.left_hand_hue_mean:.1f}")
                            
                            elif key_type_to_cal.startswith('R'):  # Right hand (RW or RB)
                                old_value = self.app_state.detection.right_hand_hue_mean
                                # Update right hand hue mean (running average if already calibrated)
                                if self.app_state.detection.right_hand_hue_mean > 0:
                                    self.app_state.detection.right_hand_hue_mean = (
                                        self.app_state.detection.right_hand_hue_mean + avg_hue) / 2
                                    self.logger.debug(f"[HUE-CALIBRATION] RIGHT hand: Updated from {old_value:.1f} to {self.app_state.detection.right_hand_hue_mean:.1f} (average with {avg_hue:.1f})")
                                else:
                                    self.app_state.detection.right_hand_hue_mean = avg_hue
                                    self.logger.debug(f"[HUE-CALIBRATION] RIGHT hand: Set initial value to {avg_hue:.1f}")
                                self.logger.info(f"Updated right hand hue mean to {self.app_state.detection.right_hand_hue_mean:.1f}")
                            else:
                                self.logger.warning(f"[HUE-CALIBRATION] WARNING: Key type '{key_type_to_cal}' doesn't start with L or R!")
                            
                            # Check if hand detection is fully calibrated
                            self.logger.debug(f"[HUE-CALIBRATION] Current state: left_hue={self.app_state.detection.left_hand_hue_mean:.1f}, right_hue={self.app_state.detection.right_hand_hue_mean:.1f}")
                            if (self.app_state.detection.left_hand_hue_mean > 0 and 
                                self.app_state.detection.right_hand_hue_mean > 0):
                                self.app_state.detection.hand_detection_calibrated = True
                                self.logger.info("Hand detection calibration complete!")
                                self.logger.info(f"[HUE-CALIBRATION] [OK] Hand detection calibration COMPLETE! Calibrated=True")
                            else:
                                self.logger.debug(f"[HUE-CALIBRATION] Hand detection not yet complete - need both left and right calibrated")
                        else:
                            self.logger.error(f"[HUE-CALIBRATION] ERROR: Could not extract ROI for hue calculation!")
                    else:
                        self.logger.debug(f"[HUE-CALIBRATION] Hand assignment disabled - skipping hue extraction")
                    
                    # Reset calibration state
                    self.app_state.calibration.calibration_mode = None
                    self.app_state.calibration.current_calibration_key_type = None
                    self.app_state.unsaved_changes = True
                    
                    # Update UI
                    if self.ui_updater:
                        self.ui_updater.update_control_panel()
                    
                    # Show success message
                    if self.ui_updater:
                        self.ui_updater.show_message(
                            "Calibration", 
                            f"Exemplar lit color for {key_type_to_cal} set to {sampled_color} from overlay."
                        )
                else:
                    # Show error message
                    if self.ui_updater:
                        self.ui_updater.show_message(
                            "Calibration Error", 
                            "Could not sample color from the selected overlay."
                        )
            
            # Always reset calibration mode after an attempt
            self.app_state.calibration.calibration_mode = None
            self.app_state.calibration.current_calibration_key_type = None

        # Update UI elements that depend on selected overlay
        if self.ui_updater:
            self.ui_updater.update_selected_overlay_display()
    
    def transpose_overlays(self, semitones: int) -> bool:
        """
        Transpose all overlays by the specified number of semitones.
        
        Args:
            semitones: Number of semitones to transpose (±12 per octave)
            
        Returns:
            bool: True if transpose was successful, False if any key would be out of range
        """
        # Calculate new transpose value
        new_transpose = self.app_state.midi.octave_transpose + (semitones // 12)
        
        # Check if all overlays would remain in valid MIDI range (0-127) with new transpose
        for overlay in self.app_state.overlays:
            # Calculate what the MIDI note would be with the new transpose
            test_midi_note = overlay.get_midi_note_number(new_transpose)
            if test_midi_note < 0 or test_midi_note > 127:
                self.logger.warning(
                    f"Cannot transpose: overlay {overlay.key_id} would produce MIDI note {test_midi_note} (out of range)"
                )
                return False
        
        # If all checks pass, update the transpose value
        self.app_state.midi.octave_transpose = new_transpose
        
        self.logger.info(f"Set octave transpose to {new_transpose}")
        
        # Mark as unsaved
        self.app_state.unsaved_changes = True
        
        # Refresh the display
        if self.ui_updater:
            self.ui_updater.refresh_canvas()
            self.ui_updater.update_control_panel()
        
        return True
    
    def adjust_overlay_sizes(self, key_color: str, dimension: str, delta: int):
        """
        Adjust overlay sizes symmetrically from center.
        
        Args:
            key_color: "white" or "black" - which key type to adjust
            dimension: "width" or "height" - which dimension to adjust
            delta: Amount to adjust (typically +2 or -2 pixels)
        """
        white_key_note_names = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
        modified_count = 0
        
        for overlay in self.app_state.overlays:
            is_white_key = overlay.note_name_in_octave in white_key_note_names
            target_is_white = key_color.lower() == "white"
            
            # Only adjust overlays of the matching key color type
            if is_white_key == target_is_white:
                if dimension == "width":
                    # Calculate new width
                    new_width = overlay.width + delta
                    if new_width < 1:  # Minimum width of 1 pixel
                        continue
                        
                    # Calculate center position
                    center_x = overlay.x + overlay.width / 2
                    
                    # Set new width and adjust x to keep center fixed
                    overlay.width = new_width
                    overlay.x = center_x - new_width / 2
                    
                    # Ensure overlay stays within image bounds
                    if overlay.x < 0:
                        overlay.x = 0
                    elif self.ui_updater and self.ui_updater.has_video_loaded():
                        video_session = self.ui_updater.get_video_session()
                        if video_session and overlay.x + overlay.width > video_session.width:
                            overlay.x = video_session.width - overlay.width
                        
                elif dimension == "height":
                    # Calculate new height
                    new_height = overlay.height + delta
                    if new_height < 1:  # Minimum height of 1 pixel
                        continue
                        
                    # Calculate center position
                    center_y = overlay.y + overlay.height / 2
                    
                    # Set new height and adjust y to keep center fixed
                    overlay.height = new_height
                    overlay.y = center_y - new_height / 2
                    
                    # Ensure overlay stays within image bounds
                    if overlay.y < 0:
                        overlay.y = 0
                    elif self.ui_updater and self.ui_updater.has_video_loaded():
                        video_session = self.ui_updater.get_video_session()
                        if video_session and overlay.y + overlay.height > video_session.height:
                            overlay.y = video_session.height - overlay.height
                
                modified_count += 1
        
        if modified_count > 0:
            self.app_state.unsaved_changes = True
            
            # Redraw frame if canvas is available
            if self.ui_updater and self.app_state.video.current_frame_index is not None:
                self.ui_updater.refresh_canvas()
            
            self.logger.info(f"Adjusted {dimension} by {delta} pixels for {modified_count} {key_color} keys")
