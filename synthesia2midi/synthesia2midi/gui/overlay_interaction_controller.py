"""Overlay interaction mode controller."""
from __future__ import annotations

import logging


class OverlayInteractionController:
    """Focused calibration controller extracted from the main window."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    def _handle_overlay_type_change(self, overlay_type: str):
        """Handle overlay type change (key/spark/shadow)."""
        logging.info(f"Overlay type changed to: {overlay_type}")

        # Update the current drawing mode in app_state
        if hasattr(self.app_state, 'ui'):
            self.app_state.ui.overlay_drawing_type = overlay_type

        # Update the canvas drawing mode if available
        if hasattr(self, 'keyboard_canvas') and self.keyboard_canvas:
            if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
                self.keyboard_canvas.interaction.overlay_drawing_type = overlay_type
                logging.info(f"Updated canvas interaction overlay type to: {overlay_type}")

        # Refresh the display to show overlay colors correctly
        if hasattr(self, 'keyboard_canvas'):
            self.keyboard_canvas.draw_overlays()
