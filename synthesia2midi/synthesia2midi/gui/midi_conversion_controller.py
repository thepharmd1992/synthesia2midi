"""MIDI conversion UI orchestration for the Qt app."""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QMessageBox

from synthesia2midi.workflows.midi_export import MidiExportService

translate = QCoreApplication.translate


class MidiConversionController:
    """Owns user-triggered MIDI conversion UI and delegates export orchestration."""

    def __init__(self, app):
        self.app = app

    def start_conversion_process(self) -> None:
        """Start MIDI conversion using the active conversion workflow."""
        app = self.app
        result = MidiExportService(app.app_state, app.conversion_workflow).export_to_default_path()

        app.control_panel.set_conversion_result(result.success, result.message)

        if result.success:
            if result.output_path is not None:
                app.midi_touchup_controller.show_conversion_complete_dialog(result.output_path)
            return

        if result.output_path is None and result.exception is None:
            QMessageBox.information(app, translate("MidiConversionController", "Error"), result.message)
            return

        title = (
            translate("MidiConversionController", "Conversion Error")
            if result.exception
            else translate("MidiConversionController", "Conversion Failed")
        )
        QMessageBox.critical(app, title, result.message)
