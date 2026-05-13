"""MIDI conversion UI orchestration for the Qt app."""
from __future__ import annotations

import datetime
import logging
import os

from PySide6.QtWidgets import QMessageBox


class MidiConversionController:
    """Owns user-triggered MIDI conversion flow and completion UI."""

    def __init__(self, app):
        self.app = app

    def start_conversion_process(self) -> None:
        """Start MIDI conversion using the active conversion workflow."""
        app = self.app
        logging.warning("[MIDI-BUTTON-CLICKED] === MIDI CONVERSION BUTTON CLICKED ===")
        logging.warning("[MIDI-BUTTON-CLICKED] User initiated MIDI conversion at %s", datetime.datetime.now())

        if not app.conversion_workflow:
            logging.error("[MIDI-BUTTON-CLICKED] FAILED: No conversion workflow available")
            QMessageBox.information(app, "Error", "Please open a video file first.")
            app.control_panel.set_conversion_result(False, "Please open a video file first.")
            return

        logging.warning("[MIDI-BUTTON-CLICKED] Conversion workflow available - proceeding with conversion")

        video_path_for_output = getattr(app.app_state.video, "original_video_path", None) or app.app_state.video.filepath
        completed_midi_dir = os.path.join(os.path.dirname(video_path_for_output), "Completed MIDI Files")
        os.makedirs(completed_midi_dir, exist_ok=True)

        video_basename = os.path.splitext(os.path.basename(video_path_for_output))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_filename = f"{video_basename}_{timestamp}.mid"
        midi_output_path = os.path.join(completed_midi_dir, midi_filename)

        logging.warning("[MIDI-CONVERSION-START] === Starting MIDI conversion process ===")
        logging.warning("[MIDI-CONVERSION-START] Output path: %s", midi_output_path)
        logging.warning("[MIDI-CONVERSION-START] Video path: %s", video_path_for_output)

        try:
            logging.warning("[MIDI-CONVERSION-START] Calling conversion_workflow.convert_to_midi()...")
            success = app.conversion_workflow.convert_to_midi(midi_output_path)
            logging.warning("[MIDI-CONVERSION-RESULT] convert_to_midi() returned: %s", success)

            if success:
                app.control_panel.set_conversion_result(True, f"MIDI file saved to:\n{midi_output_path}")
                app.midi_touchup_controller.show_conversion_complete_dialog(midi_output_path)
                logging.warning("[MIDI-CONVERSION-SUCCESS] MIDI conversion successful. Output: %s", midi_output_path)
            else:
                app.control_panel.set_conversion_result(False, "MIDI conversion failed. Check logs for details.")
                QMessageBox.critical(app, "Conversion Failed", "MIDI conversion failed. Check logs for details.")
                logging.error("[MIDI-CONVERSION-FAILED] MIDI conversion failed - convert_to_midi returned False")
        except Exception as exc:
            app.control_panel.set_conversion_result(False, f"MIDI conversion error: {str(exc)}")
            QMessageBox.critical(app, "Conversion Error", f"MIDI conversion error: {str(exc)}")
            logging.error("[MIDI-CONVERSION-EXCEPTION] MIDI conversion exception: %s", exc, exc_info=True)
