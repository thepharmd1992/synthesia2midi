"""Manual calibration wizard controller."""
from __future__ import annotations

import cv2
import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QProgressDialog, QWidget

from synthesia2midi.detection.assisted_calibration import (
    assess_unlit_frame,
    ExemplarScanSettings,
    apply_assisted_calibration_proposal,
    build_assisted_calibration_proposal,
    capture_unlit_references_from_frame,
)

from synthesia2midi.gui.auto_detect_tuning_controller import AutoDetectTuningController
from synthesia2midi.gui.assisted_calibration_dialog import (
    AssistedCalibrationDecision,
    AssistedCalibrationDialog,
)
from synthesia2midi.gui.dialog_positioning import move_to_upper_left_safe_zone

translate = QCoreApplication.translate


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
        self._manual_edit_current_calibration_requested = False
        self._pending_assisted_calibration_context: Optional[tuple[np.ndarray, int]] = None

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
        self._manual_edit_current_calibration_requested = False
        self._pending_assisted_calibration_context = None

    def _clear_calibration_wizard(self) -> None:
        self.calibration_wizard = None
        self._reset_wizard_lifecycle_flags()

    def _apply_template_styles_to_overlays(self):
        if self.calibration_workflow:
            self.calibration_workflow.apply_template_styles_to_overlays()

    def run_calibration_wizard(self):
        """Called by the Run/Reset Calibration Wizard button using CalibrationWorkflow."""
        if not self.calibration_workflow:
            QMessageBox.warning(
                self.app,
                translate("CalibrationWizardController", "Wizard Error"),
                translate("CalibrationWizardController", "Please open a video file first."),
            )
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

        edit_context_available = self._has_editable_current_calibration_context()
        edit_tooltip = (
            self._edit_current_calibration_tooltip(edit_context_available)
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
            if self._manual_edit_current_calibration_requested:
                self._clear_calibration_wizard()
                logging.info("Manual calibration edit was requested; cleaned up wizard instance")
                return
            # Keep wizard alive for modeless tuning apply callbacks.
            logging.info("Edit current calibration was requested, keeping wizard instance alive")
            return

        wizard_success = self.calibration_wizard.result is True
        manual_overlays_generated = bool(
            getattr(self.calibration_wizard, "manual_overlays_generated", False)
        )

        # Handle wizard completion
        if self.calibration_workflow.handle_wizard_completed(wizard_success):
            logging.info("Wizard submitted successfully and overlays generated.")
            if manual_overlays_generated:
                self._set_overlay_generation_source("manual")
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
        if manual_overlays_generated and self.app_state.overlays:
            manual_fit_controller = getattr(self.app, "manual_keyboard_fit_controller", None)
            if manual_fit_controller is not None:
                manual_fit_controller.open(start_setup=True)

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
            QMessageBox.warning(
                self.app,
                translate("CalibrationWizardController", "Canvas Error"),
                translate("CalibrationWizardController", "Canvas interaction system not available."),
            )
            logging.error("Canvas interaction system not available")

    def _handle_edit_current_calibration_request(self):
        """Open tuning dialog for the currently loaded calibration without redrawing ROI."""
        self._edit_current_calibration_requested = True

        if self._current_overlay_generation_source() == "manual":
            manual_fit_controller = getattr(self.app, "manual_keyboard_fit_controller", None)
            if manual_fit_controller is not None:
                opened = manual_fit_controller.open(start_setup=False)
                if opened is not False:
                    self._manual_edit_current_calibration_requested = True
                    return
            self._edit_current_calibration_requested = False
            QMessageBox.warning(
                self.app,
                translate("CalibrationWizardController", "Manual Fit"),
                translate("CalibrationWizardController", "No reusable manual calibration is available yet."),
            )
            return

        opened = self._open_auto_detect_tuning_dialog(use_wizard_context=False)
        if not opened:
            self._edit_current_calibration_requested = False
            QMessageBox.warning(
                self.app,
                translate("CalibrationWizardController", "Auto-Detect Tuning"),
                translate(
                    "CalibrationWizardController",
                    "No reusable auto-detect calibration context is available yet. Run autodetect once with ROI selection first.",
                ),
            )

    def review_current_alignment(self) -> bool:
        """Open the existing alignment editor appropriate for the loaded calibration."""
        if self._current_overlay_generation_source() == "manual":
            manual_fit_controller = getattr(self.app, "manual_keyboard_fit_controller", None)
            if manual_fit_controller is None:
                return False
            return manual_fit_controller.open(start_setup=False) is not False

        if self._has_editable_auto_detect_tuning_context():
            if self.calibration_wizard is None and self.calibration_workflow is not None:
                self._reset_wizard_lifecycle_flags()
                self.calibration_wizard = self.calibration_workflow.run_calibration_wizard()
            if self.calibration_wizard is not None:
                if self._open_auto_detect_tuning_dialog(use_wizard_context=False):
                    return True
                self._clear_calibration_wizard()

        self.run_calibration_wizard()
        return self.calibration_wizard is not None

    def run_assisted_calibration_from_current_frame(self) -> bool:
        """Use the currently displayed frame as the assisted scan baseline."""
        frame_index = int(getattr(self.app_state.video, "current_frame_index", 0) or 0)
        frame_rgb = self._frame_provider_rgb(frame_index)
        if frame_rgb is None or not isinstance(frame_rgb, np.ndarray) or frame_rgb.size == 0:
            return False
        return self._run_assisted_auto_calibration(frame_rgb, frame_index)

    def _cache_auto_detect_tuning_context(self, context: Dict[str, Any]) -> None:
        """Wizard callback adapter that keeps tuning state owned by the tuning controller."""
        self.auto_detect_tuning_controller.cache_context(context)

    def _frame_provider_rgb(self, frame_index: int):
        if not self.video_session:
            return None
        success, frame_bgr = self.video_session.get_frame(frame_index)
        if not success or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _proposal_summary_text(self, proposal) -> str:
        slot_labels = {
            "LW": translate("CalibrationWizardController", "Left White"),
            "LB": translate("CalibrationWizardController", "Left Black"),
            "RW": translate("CalibrationWizardController", "Right White"),
            "RB": translate("CalibrationWizardController", "Right Black"),
        }
        lines = [
            translate(
                "CalibrationWizardController",
                "Assisted calibration found {count} possible pressed-key samples.",
            ).format(count=proposal.candidate_count),
            translate(
                "CalibrationWizardController",
                "Found {count} Synthesia note color families.",
            ).format(count=proposal.assignment_result.family_count),
            translate(
                "CalibrationWizardController",
                "Left/Right refer to Synthesia note colors, not the physical side of the keyboard.",
            ),
        ]
        for slot in ("LW", "LB", "RW", "RB"):
            assignment = proposal.assignment_result.assignments.get(slot)
            if assignment is None:
                continue
            label = slot_labels[slot]
            if not assignment.enabled:
                lines.append(
                    translate(
                        "CalibrationWizardController",
                        "{label}: not present in this video",
                    ).format(label=label)
                )
            elif assignment.rgb is not None:
                lines.append(
                    translate(
                        "CalibrationWizardController",
                        "{label}: found",
                    ).format(label=label)
                )
            else:
                lines.append(
                    translate(
                        "CalibrationWizardController",
                        "{label}: not found",
                    ).format(label=label)
                )
        return "\n".join(lines)

    def _proposal_has_usable_assignments(self, proposal) -> bool:
        if proposal.candidate_count == 0:
            return False
        return any(
            assignment.enabled and assignment.rgb is not None
            for assignment in proposal.assignment_result.assignments.values()
        )

    def _snapshot_calibration_state(self):
        return {
            "overlays": [
                (
                    overlay,
                    overlay.unlit_reference_color,
                    overlay.unlit_hist.copy() if overlay.unlit_hist is not None else None,
                )
                for overlay in self.app_state.overlays
            ],
            "enabled": dict(self.app_state.detection.exemplar_key_type_enabled),
            "colors": copy.deepcopy(self.app_state.detection.exemplar_lit_colors),
            "histograms": {
                key: value.copy() if value is not None else None
                for key, value in self.app_state.detection.exemplar_lit_histograms.items()
            },
            "unsaved_changes": self.app_state.unsaved_changes,
        }

    def _restore_calibration_state(self, snapshot) -> None:
        for overlay, unlit_reference_color, unlit_hist in snapshot["overlays"]:
            overlay.unlit_reference_color = unlit_reference_color
            overlay.unlit_hist = unlit_hist.copy() if unlit_hist is not None else None
        self.app_state.detection.exemplar_key_type_enabled.clear()
        self.app_state.detection.exemplar_key_type_enabled.update(snapshot["enabled"])
        self.app_state.detection.exemplar_lit_colors.clear()
        self.app_state.detection.exemplar_lit_colors.update(copy.deepcopy(snapshot["colors"]))
        self.app_state.detection.exemplar_lit_histograms.clear()
        self.app_state.detection.exemplar_lit_histograms.update(
            {
                key: value.copy() if value is not None else None
                for key, value in snapshot["histograms"].items()
            }
        )
        self.app_state.unsaved_changes = snapshot["unsaved_changes"]

    def _set_assisted_calibration_guide_state(self, state: str) -> None:
        panel = getattr(self.app, "control_panel", None)
        guide = getattr(panel, "guide_page", None) if panel is not None else None
        if guide is not None and hasattr(guide, "set_assisted_state"):
            guide.set_assisted_state(state)

    def _run_assisted_auto_calibration(
        self,
        baseline_frame_rgb: np.ndarray,
        baseline_frame_index: int,
    ) -> bool:
        if (
            baseline_frame_rgb is None
            or not isinstance(baseline_frame_rgb, np.ndarray)
            or baseline_frame_rgb.size == 0
            or not self.app_state.overlays
        ):
            return False

        calibration_snapshot = self._snapshot_calibration_state()
        self._set_assisted_calibration_guide_state("scanning")
        assessment = assess_unlit_frame(baseline_frame_rgb, self.app_state.overlays)
        if assessment.should_warn:
            note_list = ", ".join(item.note_label for item in assessment.likely_lit)
            response = QMessageBox.warning(
                self.app if isinstance(self.app, QWidget) else None,
                translate("CalibrationWizardController", "Unlit Frame May Contain Lit Keys"),
                translate(
                    "CalibrationWizardController",
                    "It looks like these keys may be lit: {notes}.\n\nMove to a frame where no keys are lit, or continue if this is expected.",
                ).format(notes=note_list),
                QMessageBox.StandardButton.Ignore | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if response == QMessageBox.StandardButton.Cancel:
                self._set_assisted_calibration_guide_state("kept")
                return False
        capture_unlit_references_from_frame(baseline_frame_rgb, self.app_state.overlays)

        total_frames = getattr(self.video_session, "total_frames", baseline_frame_index + 1)
        total_frames = total_frames or (baseline_frame_index + 1)
        end_frame = max(baseline_frame_index, total_frames - 1)
        progress_parent = self.app if isinstance(self.app, QWidget) else None
        progress = QProgressDialog(
            translate("CalibrationWizardController", "Scanning for lit key examples..."),
            translate("CalibrationWizardController", "Cancel"),
            baseline_frame_index,
            end_frame,
            progress_parent,
        )
        progress.setWindowTitle(translate("CalibrationWizardController", "Assisted Calibration"))
        progress.setMinimumDuration(0)

        def progress_callback(current_frame: int, final_frame: int) -> bool:
            progress.setMaximum(final_frame)
            progress.setValue(current_frame)
            QApplication.processEvents()
            return not progress.wasCanceled()

        proposal = build_assisted_calibration_proposal(
            self._frame_provider_rgb,
            self.app_state.overlays,
            baseline_frame_index=baseline_frame_index,
            end_frame=end_frame,
            settings=ExemplarScanSettings(),
            progress_callback=progress_callback,
        )
        progress.close()
        if proposal.canceled:
            self._restore_calibration_state(calibration_snapshot)
            self._set_assisted_calibration_guide_state("kept")
            return False
        if not self._proposal_has_usable_assignments(proposal):
            self._restore_calibration_state(calibration_snapshot)
            self._set_assisted_calibration_guide_state("none_found")
            QMessageBox.information(
                self.app if isinstance(self.app, QWidget) else None,
                translate("CalibrationWizardController", "Assisted Calibration"),
                translate(
                    "CalibrationWizardController",
                    "No lit examples were found for assisted calibration. Existing calibration samples were left unchanged.",
                ),
            )
            return False

        dialog = AssistedCalibrationDialog(
            proposal,
            self.app if isinstance(self.app, QWidget) else None,
        )
        dialog.exec()
        if dialog.decision is not AssistedCalibrationDecision.USE:
            self._restore_calibration_state(calibration_snapshot)
            self._set_assisted_calibration_guide_state(
                "retry" if dialog.decision is AssistedCalibrationDecision.RETRY else "kept"
            )
            return False

        apply_assisted_calibration_proposal(self.app_state, proposal)
        for slot, assignment in proposal.assignment_result.assignments.items():
            if assignment.rgb is None:
                self.app_state.detection.exemplar_key_type_enabled[slot] = calibration_snapshot[
                    "enabled"
                ].get(slot, True)
                self.app_state.detection.exemplar_lit_colors[slot] = copy.deepcopy(
                    calibration_snapshot["colors"].get(slot)
                )
                old_histogram = calibration_snapshot["histograms"].get(slot)
                self.app_state.detection.exemplar_lit_histograms[slot] = (
                    old_histogram.copy() if old_histogram is not None else None
                )
        if self.video_loading_workflow:
            self.video_loading_workflow.save_current_config()
        self._set_assisted_calibration_guide_state("applied")
        refresh_readiness = getattr(
            getattr(self.app, "control_panel", None),
            "_update_conversion_readiness_display",
            None,
        )
        if callable(refresh_readiness):
            refresh_readiness()
        return True

    def _queue_assisted_auto_calibration(
        self,
        baseline_frame_rgb: Any,
        baseline_frame_index: int,
    ) -> None:
        if (
            baseline_frame_rgb is None
            or not isinstance(baseline_frame_rgb, np.ndarray)
            or baseline_frame_rgb.size == 0
        ):
            self._pending_assisted_calibration_context = None
            return
        self._pending_assisted_calibration_context = (
            np.copy(baseline_frame_rgb),
            int(baseline_frame_index),
        )

    def _run_pending_assisted_auto_calibration(self) -> bool:
        pending = self._pending_assisted_calibration_context
        self._pending_assisted_calibration_context = None
        if pending is None:
            return False
        baseline_frame_rgb, baseline_frame_index = pending
        return self._run_assisted_auto_calibration(
            baseline_frame_rgb,
            baseline_frame_index,
        )

    def _has_editable_auto_detect_tuning_context(self) -> bool:
        return self.auto_detect_tuning_controller.has_editable_context()

    def _current_overlay_generation_source(self) -> Optional[str]:
        calibration = getattr(self.app_state, "calibration", None)
        return getattr(calibration, "overlay_generation_source", None)

    def _set_overlay_generation_source(self, source: str) -> None:
        calibration = getattr(self.app_state, "calibration", None)
        if calibration is not None:
            calibration.overlay_generation_source = source

    def _has_editable_current_calibration_context(self) -> bool:
        if self._current_overlay_generation_source() == "manual":
            return bool(getattr(self.app_state, "overlays", []))
        return self._has_editable_auto_detect_tuning_context()

    def _edit_current_calibration_tooltip(self, enabled: bool) -> str:
        if self._current_overlay_generation_source() == "manual":
            return (
                translate("CalibrationWizardController", "Open Manual Fit for the current manual calibration.")
                if enabled
                else translate(
                    "CalibrationWizardController",
                    "Edit Current Calibration becomes available after manual overlays exist.",
                )
            )
        return (
            translate(
                "CalibrationWizardController",
                "Open the auto-detect tuning panel for the current calibration.",
            )
            if enabled
            else translate(
                "CalibrationWizardController",
                "Edit Current Calibration becomes available after an auto-detect run.",
            )
        )

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
                self._set_overlay_generation_source("auto")
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

                baseline_frame_rgb = wizard_context.get("frame_rgb") if wizard_context else None
                baseline_frame_index = self.app_state.video.current_frame_index or 0
                self._queue_assisted_auto_calibration(baseline_frame_rgb, baseline_frame_index)

                tuning_dialog_opened = self._open_auto_detect_tuning_dialog()
                if tuning_dialog_opened:
                    logging.info("Auto-detect tuning dialog opened (modeless)")
                else:
                    self._pending_assisted_calibration_context = None
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

    def _on_auto_detect_tuning_dialog_finished(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            self._run_pending_assisted_auto_calibration()
            self.control_panel.update_controls_from_state()
        else:
            self._pending_assisted_calibration_context = None

        # Cleanup wizard after modeless tuning closes.
        if self.calibration_wizard is not None:
            self._clear_calibration_wizard()
            logging.info("Cleaned up calibration wizard instance after tuning dialog closed")
