"""
Display management for Synthesia2MIDI application.

This module handles UI display toggles and view-related updates.
"""
# Standard library imports
import logging
from typing import Optional

# Local imports
from ..core.app_state import AppState
from .ui_update_interface import UIUpdateInterface


class DisplayManager:
    """Manages UI display toggles and visual updates."""
    
    def __init__(self, app_state: AppState, ui_updater: Optional[UIUpdateInterface] = None):
        """
        Initialize DisplayManager.
        
        Args:
            app_state: The application state object
            ui_updater: UI update interface for clean dependency injection
        """
        self.app_state = app_state
        self.ui_updater = ui_updater
        self.logger = logging.getLogger(f"{__name__}.DisplayManager")
    
    def toggle_overlays(self):
        """Toggle overlay visibility."""
        self.app_state.ui.show_overlays = not self.app_state.ui.show_overlays
        self.app_state.unsaved_changes = True  # This is a view setting, but good to save preference
        
        # Update UI if updater is available
        if self.ui_updater:
            self.ui_updater.update_overlay_action(self.app_state.ui.show_overlays)
            
            # Redraw frame if video is loaded
            if self.app_state.video.current_frame_index is not None:
                self.ui_updater.refresh_canvas()
        
        self.logger.info(f"Overlays {'shown' if self.app_state.ui.show_overlays else 'hidden'}")

    def toggle_live_detection_feedback(self):
        """Toggle live detection feedback on the canvas."""
        self.app_state.ui.live_detection_feedback = not self.app_state.ui.live_detection_feedback
        self.app_state.unsaved_changes = True  # Save this preference
        
        # Update UI if updater is available
        if self.ui_updater:
            self.ui_updater.update_live_detection_action(self.app_state.ui.live_detection_feedback)
            
            # Force a redraw to update overlay colors
            if self.ui_updater.has_video_loaded():
                self.ui_updater.refresh_canvas()
        
        self.logger.info(f"Live detection feedback {'enabled' if self.app_state.ui.live_detection_feedback else 'disabled'}")

    def handle_refresh_selected_overlay_display(self):
        """Callback to explicitly refresh the selected overlay display in ControlPanel."""
        if self.ui_updater:
            self.ui_updater.update_selected_overlay_display()

    def handle_visual_threshold_monitor_toggle(self, enabled: bool):
        """Handle visual threshold monitor enable/disable."""
        self.app_state.ui.visual_threshold_monitor_enabled = enabled
        self.app_state.unsaved_changes = True
        self.logger.info(f"Visual threshold monitor {'enabled' if enabled else 'disabled'}")
        
        # Force a redraw to update or hide the debug box
        if self.ui_updater and self.ui_updater.has_video_loaded():
            self.ui_updater.refresh_canvas()
