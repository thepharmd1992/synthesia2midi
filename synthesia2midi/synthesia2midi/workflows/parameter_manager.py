"""
Simple parameter management for Synthesia2MIDI application.

This module handles simple parameter updates like tempo, navigation interval,
and testing mode.
"""

import logging

from PySide6.QtWidgets import QInputDialog

from ..core.app_state import AppState
from ..core.state_manager import StateManager


class ParameterManager:
    """Manages simple parameter updates for tempo, navigation, and testing mode."""
    
    def __init__(self, app_state: AppState, state_manager: StateManager, parent_widget=None):
        """
        Initialize ParameterManager.
        
        Args:
            app_state: The application state object
            state_manager: The state manager for tempo updates
            parent_widget: Parent widget for dialogs (optional)
        """
        self.app_state = app_state
        self.state_manager = state_manager
        self.parent_widget = parent_widget
    
    def update_tempo(self, value: int):
        """Update tempo via state manager."""
        self.state_manager.update_tempo(value)
        logging.debug(f"Tempo updated to: {value}")
    
    def update_nav_interval(self, value: int):
        """Update navigation interval."""
        self.app_state.video.current_nav_interval = value
        self.app_state.unsaved_changes = True  # Arguably not a "setting" but part of UI state to persist
        logging.debug(f"Frame navigation interval: {value}")
    
