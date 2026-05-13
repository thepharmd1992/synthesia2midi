"""Manual calibration wizard and auto-detect tuning controller."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog


class CalibrationWizardController:
    """Extracted controller for calibration-related main-window behavior."""

    def __init__(self, app):
        self.app = app
        self.calibration_wizard = None
        self._auto_detect_tuning_dialog = None
        self._last_auto_detect_tuning_context: Optional[Dict[str, Any]] = None

    def __getattr__(self, name):
        return getattr(self.app, name)

    def _invoke_calibration_wizard(self):
        """Called by the Run/Reset Calibration Wizard button using CalibrationWorkflow."""
        if not self.calibration_workflow:
            QMessageBox.warning(self.app, "Wizard Error", "Please open a video file first.")
            # Ensure wizard button is disabled if no video session somehow
            if hasattr(self.control_panel, 'wizard_button'):
                 self.control_panel.wizard_button.setEnabled(False)
            return

        logging.info("Starting manual calibration wizard invocation.")

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

        # Mark that keyboard region was not requested yet
        self.calibration_wizard._keyboard_region_requested = False
        self.calibration_wizard._edit_current_calibration_requested = False
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
        result = self.calibration_wizard.exec()

        # Check if keyboard region selection was requested
        if hasattr(self.calibration_wizard, '_keyboard_region_requested') and self.calibration_wizard._keyboard_region_requested:
            # Don't cleanup wizard yet - we need it for keyboard region selection
            logging.info("Keyboard region selection was requested, keeping wizard instance alive")
            return
        if hasattr(self.calibration_wizard, '_edit_current_calibration_requested') and self.calibration_wizard._edit_current_calibration_requested:
            # Keep wizard alive for modeless tuning apply callbacks.
            logging.info("Edit current calibration was requested, keeping wizard instance alive")
            return

        wizard_success = self.calibration_wizard.result is True

        # Handle wizard completion
        if self.calibration_workflow.handle_wizard_completed(wizard_success):
            logging.info("Wizard submitted successfully and overlays generated.")
            # Apply template styles (kept in main class for UI coordination)
            self._apply_template_styles_to_overlays()


            self.control_panel.convert_button.setEnabled(self.control_panel._can_convert())
            self.keyboard_canvas.draw_overlays() # Explicitly redraw overlays
        else:
            logging.info("Wizard was cancelled or did not generate overlays. Convert button remains disabled.")
            self.control_panel.convert_button.setEnabled(False)

        # Cleanup
        self.calibration_wizard = None

        # Refresh UI elements
        self.keyboard_canvas.display_frame(self.app_state.video.current_frame_index) # Redraw frame and overlays

    def _handle_keyboard_region_selection_request(self):
        """Handle request to select keyboard region from wizard."""
        logging.info("Starting keyboard region selection mode")

        # Mark that keyboard region was requested
        if self.calibration_wizard:
            self.calibration_wizard._keyboard_region_requested = True

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
        if self.calibration_wizard:
            self.calibration_wizard._edit_current_calibration_requested = True

        opened = self._open_auto_detect_tuning_dialog(use_wizard_context=False)
        if not opened:
            if self.calibration_wizard:
                self.calibration_wizard._edit_current_calibration_requested = False
            QMessageBox.warning(
                self.app,
                "Auto-Detect Tuning",
                "No reusable auto-detect calibration context is available yet. "
                "Run autodetect once with ROI selection first.",
            )

    def _clone_auto_detect_tuning_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = dict(context)
        frame_rgb = context.get("frame_rgb")
        if frame_rgb is not None:
            cloned["frame_rgb"] = np.copy(frame_rgb)
        keyboard_roi = context.get("keyboard_roi")
        if keyboard_roi is not None:
            cloned["keyboard_roi"] = tuple(int(v) for v in keyboard_roi)
        cloned["fallback_used"] = bool(context.get("fallback_used", False))
        return cloned

    def _cache_auto_detect_tuning_context(self, context: Dict[str, Any]) -> None:
        self._last_auto_detect_tuning_context = self._clone_auto_detect_tuning_context(context)

    def _get_current_frame_rgb_for_tuning(self) -> Optional[np.ndarray]:
        frame_rgb = getattr(self.keyboard_canvas, "current_frame_rgb", None)
        if frame_rgb is not None:
            return np.copy(frame_rgb)

        frame_idx = self.app_state.video.current_frame_index
        if self.video_session is None or frame_idx is None:
            return None

        frame_bgr = self.video_session.get_frame(frame_idx)
        if frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _build_auto_detect_tuning_context_from_state(self) -> Optional[Dict[str, Any]]:
        if not self.app_state.overlays:
            return None

        frame_rgb = self._get_current_frame_rgb_for_tuning()
        if frame_rgb is None:
            return None

        valid_overlays = [
            overlay
            for overlay in self.app_state.overlays
            if overlay.width > 0 and overlay.height > 0
        ]
        if not valid_overlays:
            return None

        min_x = min(float(overlay.x) for overlay in valid_overlays)
        min_y = min(float(overlay.y) for overlay in valid_overlays)
        max_x = max(float(overlay.x) + float(overlay.width) for overlay in valid_overlays)
        max_y = max(float(overlay.y) + float(overlay.height) for overlay in valid_overlays)

        frame_h, frame_w = frame_rgb.shape[:2]
        x = max(0, int(np.floor(min_x)))
        y = max(0, int(np.floor(min_y)))
        right = min(frame_w, int(np.ceil(max_x)))
        bottom = min(frame_h, int(np.ceil(max_y)))
        width = right - x
        height = bottom - y
        if width <= 0 or height <= 0:
            return None

        detection_results: Dict[str, Any] = {
            "total_keys": int(self.app_state.midi.total_keys),
            "leftmost_note": self.app_state.midi.leftmost_note_name,
            "leftmost_octave": int(self.app_state.midi.leftmost_note_octave),
            "detected_keys": [],
        }
        if self._last_auto_detect_tuning_context is not None:
            cached_results = self._last_auto_detect_tuning_context.get("detection_results")
            if isinstance(cached_results, dict):
                detection_results = dict(cached_results)

        return {
            "frame_rgb": frame_rgb,
            "keyboard_roi": (x, y, width, height),
            "fallback_used": bool(
                self._last_auto_detect_tuning_context.get("fallback_used", False)
                if self._last_auto_detect_tuning_context is not None
                else False
            ),
            "detection_results": detection_results,
        }

    def _resolve_auto_detect_tuning_context(self, *, use_wizard_context: bool) -> Optional[Dict[str, Any]]:
        if use_wizard_context and self.calibration_wizard:
            wizard_context = self.calibration_wizard.get_auto_detect_tuning_context()
            if wizard_context:
                self._cache_auto_detect_tuning_context(wizard_context)
                return self._clone_auto_detect_tuning_context(wizard_context)

        if self._last_auto_detect_tuning_context is not None:
            return self._clone_auto_detect_tuning_context(self._last_auto_detect_tuning_context)

        state_context = self._build_auto_detect_tuning_context_from_state()
        if state_context is not None:
            self._cache_auto_detect_tuning_context(state_context)
            return self._clone_auto_detect_tuning_context(state_context)

        if not use_wizard_context and self.calibration_wizard:
            wizard_context = self.calibration_wizard.get_auto_detect_tuning_context()
            if wizard_context:
                self._cache_auto_detect_tuning_context(wizard_context)
                return self._clone_auto_detect_tuning_context(wizard_context)

        return None

    def _has_editable_auto_detect_tuning_context(self) -> bool:
        if self._last_auto_detect_tuning_context is not None:
            return True
        if not self.app_state.overlays:
            return False
        return self._get_current_frame_rgb_for_tuning() is not None

    def _handle_keyboard_region_selected(self, x: int, y: int, width: int, height: int):
        """Handle the keyboard region selection from canvas."""
        logging.info(f"=== KEYBOARD REGION SELECTION RECEIVED IN MAIN APP ===")
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
                self.calibration_wizard = None
                logging.info("Cleaned up calibration wizard instance")
        except Exception as e:
            logging.error(f"Error calling wizard's keyboard region handler: {e}", exc_info=True)

        self.control_panel.update_controls_from_state() # Reflect any changes
        self.control_panel.update_trim_controls_from_state() # Update frame range controls
        self.control_panel.update_selected_overlay_display() # Refresh selected overlay info

    def _apply_auto_detect_preview_result(self, detection_results: Dict[str, Any]) -> bool:
        if not self.calibration_wizard:
            return False

        applied = self.calibration_wizard.apply_auto_detect_results(detection_results)
        if not applied:
            return False

        wizard_context = self.calibration_wizard.get_auto_detect_tuning_context()
        if wizard_context:
            self._cache_auto_detect_tuning_context(wizard_context)
        elif self._last_auto_detect_tuning_context is not None:
            self._last_auto_detect_tuning_context["detection_results"] = dict(detection_results)

        self._apply_template_styles_to_overlays()
        self.app_state.ui.show_overlays = True
        self.show_overlays_action.setChecked(True)
        self.control_panel.convert_button.setEnabled(self.control_panel._can_convert())

        current_frame = self.app_state.video.current_frame_index
        if current_frame is not None:
            self.keyboard_canvas.display_frame(current_frame)
        else:
            self.keyboard_canvas.update()

        self.control_panel.update_controls_from_state()
        self.control_panel.update_selected_overlay_display()
        return True

    def _open_auto_detect_tuning_dialog(self, *, use_wizard_context: bool = True) -> bool:
        if not self.calibration_wizard:
            return False

        context = self._resolve_auto_detect_tuning_context(use_wizard_context=use_wizard_context)
        if not context:
            logging.warning("Missing auto-detect tuning context; skipping tuning dialog")
            return False

        if self._auto_detect_tuning_dialog is not None:
            try:
                self._auto_detect_tuning_dialog.finished.disconnect(self._on_auto_detect_tuning_dialog_finished)
            except Exception:
                pass
            self._auto_detect_tuning_dialog.close()
            self._auto_detect_tuning_dialog = None

        dialog = AutoDetectTuningDialog(
            self.app,
            self.app_state,
            context["frame_rgb"],
            context["keyboard_roi"],
            initial_detection_results=context.get("detection_results"),
            fallback_used=bool(context.get("fallback_used", False)),
            apply_detection_callback=self._apply_auto_detect_preview_result,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.finished.connect(self._on_auto_detect_tuning_dialog_finished)
        self._auto_detect_tuning_dialog = dialog

        # Position dialog toward the right side so it does not cover the keyboard area.
        screen = self.screen() if hasattr(self, "screen") else None
        if screen is not None:
            available = screen.availableGeometry()
            frame = dialog.frameGeometry()
            x = max(
                available.left() + 10,
                available.right() - frame.width() - 20,
            )
            y = min(
                max(available.top() + 40, self.geometry().top() + 20),
                max(available.top() + 10, available.bottom() - frame.height() - 20),
            )
            dialog.move(x, y)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def _on_auto_detect_tuning_dialog_finished(self, _result: int) -> None:
        self._auto_detect_tuning_dialog = None

        # Persist tuned params/overlays with the existing per-video save flow.
        if self.app_state.unsaved_changes and self.video_loading_workflow:
            self.video_loading_workflow.save_current_config()

        # Cleanup wizard after modeless tuning closes.
        if self.calibration_wizard is not None:
            self.calibration_wizard = None
            logging.info("Cleaned up calibration wizard instance after tuning dialog closed")
