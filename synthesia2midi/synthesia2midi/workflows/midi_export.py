"""High-level MIDI export orchestration."""
from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
from pathlib import Path
from typing import Protocol

from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


class MidiConversionWorkflow(Protocol):
    """Protocol for workflows that can render MIDI to a target path."""

    def convert_to_midi(self, output_path: str) -> bool:
        """Convert the active video/session state to a MIDI file."""
        ...


@dataclass(frozen=True)
class MidiExportResult:
    """Result of a user-triggered MIDI export attempt."""

    success: bool
    message: str
    output_path: str | None = None
    exception: Exception | None = None


class MidiExportService:
    """Owns output-path selection and conversion workflow invocation for MIDI export."""

    def __init__(
        self,
        app_state,
        conversion_workflow: MidiConversionWorkflow | None,
        runtime_paths: RuntimePaths | None = None,
    ):
        self.app_state = app_state
        self.conversion_workflow = conversion_workflow
        self.runtime_paths = runtime_paths or detect_runtime_paths()
        self.logger = logging.getLogger(f"{__name__}.MidiExportService")

    def export_to_default_path(self) -> MidiExportResult:
        """Export the current video to the default user-facing MIDI output folder."""
        self.logger.warning("[MIDI-BUTTON-CLICKED] === MIDI CONVERSION BUTTON CLICKED ===")
        self.logger.warning("[MIDI-BUTTON-CLICKED] User initiated MIDI conversion at %s", datetime.datetime.now())

        if not self.conversion_workflow:
            self.logger.error("[MIDI-BUTTON-CLICKED] FAILED: No conversion workflow available")
            return MidiExportResult(False, "Please open a video file first.")

        self.logger.warning("[MIDI-BUTTON-CLICKED] Conversion workflow available - proceeding with conversion")

        try:
            midi_output_path = self._build_default_output_path()
            self.logger.warning("[MIDI-CONVERSION-START] === Starting MIDI conversion process ===")
            self.logger.warning("[MIDI-CONVERSION-START] Output path: %s", midi_output_path)
            self.logger.warning("[MIDI-CONVERSION-START] Video path: %s", self._video_path_for_output())
            self.logger.warning("[MIDI-CONVERSION-START] Calling conversion_workflow.convert_to_midi()...")

            success = self.conversion_workflow.convert_to_midi(midi_output_path)
            self.logger.warning("[MIDI-CONVERSION-RESULT] convert_to_midi() returned: %s", success)

            if success:
                self.logger.warning("[MIDI-CONVERSION-SUCCESS] MIDI conversion successful. Output: %s", midi_output_path)
                return MidiExportResult(True, f"MIDI file saved to:\n{midi_output_path}", midi_output_path)

            self.logger.error("[MIDI-CONVERSION-FAILED] MIDI conversion failed - convert_to_midi returned False")
            return MidiExportResult(False, "MIDI conversion failed. Check logs for details.", midi_output_path)
        except Exception as exc:
            self.logger.error("[MIDI-CONVERSION-EXCEPTION] MIDI conversion exception: %s", exc, exc_info=True)
            return MidiExportResult(False, f"MIDI conversion error: {str(exc)}", exception=exc)

    def _video_path_for_output(self) -> str:
        video_state = self.app_state.video
        return getattr(video_state, "original_video_path", None) or video_state.filepath

    def _build_default_output_path(self) -> str:
        video_path_for_output = self._video_path_for_output()
        completed_midi_dir = self.runtime_paths.midi_exports_dir()
        completed_midi_dir.mkdir(parents=True, exist_ok=True)

        video_basename = Path(video_path_for_output).stem or "synthesia2midi"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_filename = f"{video_basename}_{timestamp}.mid"
        return str(completed_midi_dir / midi_filename)
