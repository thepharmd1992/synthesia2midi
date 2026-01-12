"""
UI Update Interface for Synthesia2MIDI application.

This module defines the interface for UI updates, allowing workflow modules
to communicate with the UI without direct dependencies on main.py.
"""
# Standard library imports
from typing import Optional


class UIUpdateInterface:
    """Interface for UI update operations."""
    
    def update_overlay_action(self, checked: bool) -> None:
        """Update the overlay visibility action state."""
        raise NotImplementedError("Subclass must implement update_overlay_action")
    
    def refresh_canvas(self) -> None:
        """Refresh the keyboard canvas display."""
        raise NotImplementedError("Subclass must implement refresh_canvas")
    
    def update_control_panel(self) -> None:
        """Update the control panel display."""
        raise NotImplementedError("Subclass must implement update_control_panel")
    
    def update_selected_overlay_display(self) -> None:
        """Update the selected overlay display in control panel."""
        raise NotImplementedError("Subclass must implement update_selected_overlay_display")
    
    
    def update_live_detection_action(self, checked: bool) -> None:
        """Update live detection action state."""
        raise NotImplementedError("Subclass must implement update_live_detection_action")
    
    def update_detection_threshold(self, value: float) -> None:
        """Update detection threshold spinner value."""
        raise NotImplementedError("Subclass must implement update_detection_threshold")
    
    def show_message(self, title: str, message: str) -> None:
        """Show a message to the user."""
        raise NotImplementedError("Subclass must implement show_message")
    
    def get_video_session(self) -> Optional[object]:
        """Get current video session if available."""
        raise NotImplementedError("Subclass must implement get_video_session")
    
    def has_video_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        raise NotImplementedError("Subclass must implement has_video_loaded")
    
    def get_total_frames(self) -> Optional[int]:
        """Get total frames in current video."""
        raise NotImplementedError("Subclass must implement get_total_frames")
    
    def get_roi_bgr(self, overlay: object) -> Optional[object]:
        """Get ROI BGR from keyboard canvas for given overlay."""
        raise NotImplementedError("Subclass must implement get_roi_bgr")