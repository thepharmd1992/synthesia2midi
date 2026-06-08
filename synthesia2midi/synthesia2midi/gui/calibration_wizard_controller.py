"""Manual calibration wizard controller."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.gui.auto_detect_tuning_controller import AutoDetectTuningController
from synthesia2midi.gui.dialog_positioning import move_to_upper_left_safe_zone


class CalibrationWizardController:
    """Owns manual calibration wizard lifecycle and delegates tuning-dialog flow."""

    def __init__(self, app, auto_detect_tuning_controller: Optional[AutoDetectTuningController] = None):
        self.app = app
        self.calibration_wizard = None
        self.auto_detect_tuning_controller = auto_detect_tuning_controller or AutoDetectTuningController(app)
        self.auto_detect_tuning_controller.set_apply_template_styles_callback(
            self._apply_template_styles_to_overlays
        )
        self._keyboard_region_requested = False
        self._edit_current_calibration_requested = False

    @property
    def app_state(self):
        return self.app.app_state

    @property
    def calibration_workflow(self):
        return self.app.calibration_workflow

    @property
    def control_panel(self):
        return self.app.control_panel

    @property
    def keyboard_canvas(self):
        return self.app.keyboard_canvas

    @property
    def show_overlays_action(self):
        return self.app.show_overlays_action

    @property
    def video_loading_workflow(self):
        return self.app.video_loading_workflow

    @property
    def video_session(self):
        return self.app.video_session

    def _reset_wizard_lifecycle_flags(self) -> None:
        self._keyboard_region_requested = False
        self._edit_current_calibration_requested = False

    def _clear_calibration_wizard(self) -> None:
        self.calibration_wizard = None
        self._reset_wizard_lifecycle_flags()

    def _apply_template_styles_to_overlays(self):
        if self.calibration_workflow:
            self.calibration_workflow.apply_template_styles_to_overlays()

    def run_calibration_wizard(self):
        """Called by the Run/Reset Calibration Wizard button using CalibrationWorkflow."""
        if not self.calibration_workflow:
            QMessageBox.warning(self.app, "Wizard Error", "Please open a video file first.")
            # Ensure wizard button is disabled if no video session somehow
            if hasattr(self.control_panel, 'wizard_button'):
                 self.control_panel.wizard_button.setEnabled(False)
            return

        logging.info("Starting manual calibration wizard invocation.")
        self._reset_wizard_lifecycle_flags()

        # Use CalibrationWorkflow to create the wizard
        self.calibration_wizard = self.calibration_workflow.run_calibration_wizard()

        if not self.calibration_wizard:
            logging.error("Failed to create calibration wizard")
            return

        # Connect signals for keyboard region selection
        self.calibration_wizard.keyboard_region_selection_requested.connect(
            self._handle_keyboard_region_selection_request
        )
        self.calibration_wizard.edit_current_calibration_requested.connect(
            self._handle_edit_current_calibration_request
        )

        edit_context_available = self._has_editable_auto_detect_tuning_context()
        edit_tooltip = (
            "Open the auto-detect tuning panel for the current calibration."
            if edit_context_available
            else "Edit Current Calibration becomes available after an auto-detect run."
        )
        self.calibration_wizard.set_edit_current_calibration_enabled(
            edit_context_available,
            tooltip=edit_tooltip,
        )

        # Show the wizard
        move_to_upper_left_safe_zone(self.calibration_wizard, self.app)
        self.calibration_wizard.exec()

        # Check if keyboard region selection was requested
        if self._keyboard_region_requested:
            # Don't cleanup wizard yet - we need it for keyboard region selection
            logging.info("Keyboard region selection was requested, keeping wizard instance alive")
            return
        if self._edit_current_calibration_requested:
            # Keep wizard alive for modeless tuning apply callbacks.
            logging.info("Edit current calibration was requested, keeping wizard instance alive")
            return

        wizard_success = self.calibration_wizard.result is True

        # Handle wizard completion
        if self.calibration_workflow.handle_wizard_completed(wizard_success):
            logging.info("Wizard submitted successfully and overlays generated.")
            # Apply template styles (kept in main class for UI coordination)
            self._apply_template_styles_to_overlays()

            self.app_state.ui.show_overlays = True
            self.show_overlays_action.setChecked(True)
            self.control_panel.convert_button.setEnabled(self.control_panel._can_convert())
            self.keyboard_canvas.draw_overlays() # Explicitly redraw overlays
        else:
            logging.info("Wizard was cancelled or did not generate overlays. Convert button remains disabled.")
            self.control_panel.convert_button.setEnabled(False)

        # Cleanup
        self._clear_calibration_wizard()

        # Refresh UI elements
        self.keyboard_canvas.display_frame(self.app_state.video.current_frame_index) # Redraw frame and overlays

    def _handle_keyboard_region_selection_request(self):
        """Wizard signal callback; kept as a named slot for Qt signal wiring."""
        logging.info("Starting keyboard region selection mode")

        # Mark that keyboard region was requested
        self._keyboard_region_requested = True

        # Connect the keyboard region selected signal
        if hasattr(self.keyboard_canvas, 'interaction') and self.keyboard_canvas.interaction:
            logging.info("Canvas interaction available, setting up signal connection")

            # Disconnect any previous connections
            try:
                # Check if signal has any connections before disconnecting
                if hasattr(self.keyboard_canvas.interaction.keyboard_region_selected, '__self__'):
                    self.keyboard_canvas.interaction.keyboard_region_selected.disconnect()
                    logging.debug("Disconnected previous keyboard_region_selected connections")
                else:
                    logging.debug("No connections to disconnect")
            except (TypeError, RuntimeError):
                # No connections to disconnect, which is fine
                logging.debug("No previous connections to disconnect")
                pass

            # Connect to our handler
            logging.info("Connecting keyboard_region_selected signal to _handle_keyboard_region_selected")
            self.keyboard_canvas.interaction.keyboard_region_selected.connect(
                self._handle_keyboard_region_selected
            )
            logging.info("Signal connected successfully")

            # Enter selection mode
            logging.info("Entering keyboard region selection mode")
            self.keyboard_canvas.interaction.enter_keyboard_region_selection_mode()
            self.keyboard_canvas.setCursor(Qt.CrossCursor)
            logging.info("Selection mode activated with crosshair cursor")
        else:
            QMessageBox.warning(self.app, "Canvas Error", "Canvas interaction system not available.")
            logging.error("Canvas interaction system not available")

    def _handle_edit_current_calibration_request(self):
        """Open tuning dialog for the currently loaded calibration without redrawing ROI."""
        self._edit_current_calibration_requested = True

        opened = self._open_auto_detect_tuning_dialog(use_wizard_context=False)
        if not opened:
            self._edit_current_calibration_requested = False
            QMessageBox.warning(
                self.app,
                "Auto-Detect Tuning",
                "No reusable auto-detect calibration context is available yet. "
                "Run autodetect once with ROI selection first.",
            )

    def _cache_auto_detect_tuning_context(self, context: Dict[str, Any]) -> None:
        """Wizard callback adapter that keeps tuning state owned by the tuning controller."""
        self.auto_detect_tuning_controller.cache_context(context)

    def _has_editable_auto_detect_tuning_context(self) -> bool:
        return self.auto_detect_tuning_controller.has_editable_context()

    def _handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int):
        """Handle the keyboard region selection from canvas."""
        logging.info("=== KEYBOARD REGION SELECTION RECEIVED IN MAIN APP ===")
        logging.info(f"User-drawn ROI rectangle coordinates: x={x}, y={y}, width={width}, height={height}")
        logging.info(f"ROI rectangle bounds: left={x}, right={x+width}, top={y}, bottom={y+height}")
        logging.info("This ROI rectangle will be used to crop the frame for auto-detection")

        # Reset cursor
        self.keyboard_canvas.setCursor(Qt.ArrowCursor)
        logging.debug("Reset cursor to arrow")

        if not self.calibration_wizard:
            logging.error("No calibration wizard available to handle keyboard region selection")
            logging.error(f"calibration_wizard is: {self.calibration_wizard}")
            return

        logging.info("Calibration wizard exists, calling handle_keyboard_region_selected")
        try:
            tuning_dialog_opened = False
            detection_success = self.calibration_wizard.handle_keyboard_region_selected(x, y, width, height)
            if detection_success and self.app_state.overlays:
                logging.info(f"Successfully created {len(self.app_state.overlays)} overlays")
                wizard_context = self.calibration_wizard.get_auto_detect_tuning_context()
                if wizard_context:
                    self._cache_auto_detect_tuning_context(wizard_context)

                self._apply_template_styles_to_overlays()
                self.app_state.ui.show_overlays = True
                self.show_overlays_action.setChecked(True)
                self.control_panel.convert_button.setEnabled(self.control_panel._can_convert())

                current_frame = self.app_state.video.current_frame_index
                if current_frame is not None:
                    self.keyboard_canvas.display_frame(current_frame)
                else:
                    self.keyboard_canvas.update()

                tuning_dialog_opened = self._open_auto_detect_tuning_dialog()
                if tuning_dialog_opened:
                    logging.info("Auto-detect tuning dialog opened (modeless)")
                else:
                    logging.info("Auto-detect tuning dialog not opened; wizard will be cleaned up")
            else:
                logging.info("Auto-detect did not produce overlays; skipping tuning dialog")

            # For modeless tuning, keep wizard alive until tuning closes.
            if not tuning_dialog_opened:
                self._clear_calibration_wizard()
                logging.info("Cleaned up calibration wizard instance")
        except Exception as e:
            logging.error(f"Error calling wizard's keyboard region handler: {e}", exc_info=True)

        self.control_panel.update_controls_from_state() # Reflect any changes
        self.control_panel.update_trim_controls_from_state() # Update frame range controls
        self.control_panel.update_selected_overlay_display() # Refresh selected overlay info

    def _open_auto_detect_tuning_dialog(self, *, use_wizard_context: bool = True) -> bool:
        return self.auto_detect_tuning_controller.open(
            self.calibration_wizard,
            use_wizard_context=use_wizard_context,
            on_dialog_finished=self._on_auto_detect_tuning_dialog_finished,
            restore_settings_on_finish=True,
        )

    def _on_auto_detect_tuning_dialog_finished(self) -> None:
        # Cleanup wizard after modeless tuning closes.
        if self.calibration_wizard is not None:
            self._clear_calibration_wizard()
            logging.info("Cleaned up calibration wizard instance after tuning dialog closed")
