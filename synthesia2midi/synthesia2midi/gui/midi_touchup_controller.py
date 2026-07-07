"""Rust MIDI touch-up editor integration for the Qt app."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

from PySide6.QtCore import QCoreApplication, QObject, QProcess, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMessageBox

from synthesia2midi.runtime_paths import detect_runtime_paths

translate = QCoreApplication.translate


class MidiTouchupController(QObject):
    """Owns the Rust MIDI touch-up editor process lifecycle."""

    editor_started = Signal(str, str)  # source MIDI path, editor binary path
    editor_saved = Signal(str, str)  # source MIDI path, saved MIDI path shown to the user
    editor_cancelled = Signal(str)  # source MIDI path
    editor_failed = Signal(str, str)  # source MIDI path, user-facing failure message
    setup_required = Signal(str)  # source MIDI path that needs the Rust editor built first

    def __init__(self, app):
        parent = app if isinstance(app, QObject) else None
        super().__init__(parent)
        self.app = app
        # QProcess objects must be strongly retained until finished/destroyed; otherwise
        # Qt can destroy the process while the Rust editor is still running.
        self.processes: list[QProcess] = []

    def show_conversion_complete_dialog(self, midi_output_path: str) -> None:
        app = self.app
        msg_box = QMessageBox(app)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(translate("MidiTouchupController", "Conversion Complete"))
        msg_box.setText(
            translate("MidiTouchupController", "MIDI file saved to:\n{midi_output_path}").format(
                midi_output_path=midi_output_path
            )
        )
        msg_box.setInformativeText(
            translate(
                "MidiTouchupController",
                "You can open the Touch-Up Editor now, or show the saved MIDI in its folder.",
            )
        )
        open_btn = msg_box.addButton(translate("MidiTouchupController", "Open Touch-Up Editor"), QMessageBox.ActionRole)
        show_folder_btn = msg_box.addButton(
            translate("MidiTouchupController", "Show MIDI in Folder"),
            QMessageBox.AcceptRole,
        )
        msg_box.setDefaultButton(open_btn)
        msg_box.exec()

        clicked_button = msg_box.clickedButton()
        if clicked_button is open_btn:
            self.open_editor(midi_output_path)
        elif clicked_button is show_folder_btn:
            self._show_midi_in_folder(midi_output_path)

    def _show_midi_in_folder(self, midi_path: str) -> None:
        path = Path(midi_path)
        if sys.platform == "darwin":
            subprocess.run(["open", "-R", str(path)], check=False)
            return
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", f"/select,{os.path.normpath(path)}"], check=False)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def open_from_picker(self) -> None:
        runtime_paths = detect_runtime_paths()
        videos_dir = runtime_paths.default_video_dir()
        start_dir = str(videos_dir if videos_dir.exists() else runtime_paths.home_dir)

        midi_path, _ = QFileDialog.getOpenFileName(
            self.app,
            translate("MidiTouchupController", "Open MIDI for Touch-Up"),
            start_dir,
            translate("MidiTouchupController", "MIDI Files (*.mid *.midi)"),
        )
        if not midi_path:
            return
        self.open_editor(midi_path)

    def open_editor(self, midi_path: str) -> None:
        app = self.app
        if not os.path.exists(midi_path):
            message = translate("MidiTouchupController", "MIDI file not found:\n{midi_path}").format(
                midi_path=midi_path
            )
            self.editor_failed.emit(midi_path, message)
            QMessageBox.warning(app, translate("MidiTouchupController", "Touch-Up Editor"), message)
            return

        binary_path = self.resolve_binary_path()
        if binary_path is None:
            self.setup_required.emit(midi_path)
            self.show_setup_dialog(midi_path)
            return

        process = QProcess(app)
        process.setProgram(binary_path)
        process.setArguments(["--midi", midi_path, "--result-json", "--theme", "neothesia"])
        process.setProcessChannelMode(QProcess.SeparateChannels)
        # Preserve the historical launch contract: inherit the app environment and
        # current working directory rather than forcing editor-specific overrides.
        self._retain_process(process)

        def _on_destroyed(_obj=None, proc=process):
            self.remove_process_ref(proc)

        def _on_finished(exit_code: int, _exit_status: QProcess.ExitStatus, proc=process, src_path=midi_path):
            self.handle_process_finished(proc, src_path, exit_code)

        process.destroyed.connect(_on_destroyed)
        process.finished.connect(_on_finished)

        process.start()
        if not process.waitForStarted(2000):
            error_msg = process.errorString() or "Unknown launch failure."
            self.cleanup_process(process)
            if not app._is_closing:
                self.editor_failed.emit(midi_path, error_msg)
                QMessageBox.critical(
                    app,
                    translate("MidiTouchupController", "Touch-Up Editor Launch Failed"),
                    translate(
                        "MidiTouchupController",
                        "Failed to start Rust touch-up editor.\n\nBinary: {binary_path}\nError: {error_msg}",
                    ).format(binary_path=binary_path, error_msg=error_msg),
                )
            return

        self.editor_started.emit(midi_path, binary_path)

    def resolve_binary_path(self) -> Optional[str]:
        binary_path = detect_runtime_paths().rust_editor_path()
        return str(binary_path) if binary_path is not None else None

    def show_setup_dialog(self, midi_path: str) -> None:
        runtime_paths = detect_runtime_paths()
        if runtime_paths.frozen:
            QMessageBox.warning(
                self.app,
                translate("MidiTouchupController", "Touch-Up Editor Missing"),
                translate(
                    "MidiTouchupController",
                    "Bundled Rust touch-up editor files were not found.\n\nMIDI requested: {midi_path}\n\nRe-download the app build or use the repository developer setup if you are running from source.",
                ).format(midi_path=midi_path),
            )
            return

        repo_root = str(runtime_paths.repo_root)
        setup_cmd = "py setup_env.py" if os.name == "nt" else "python3 setup_env.py"
        expected_rel = os.path.join(
            "tools",
            "midi_touchup_editor_rust",
            "target",
            "release",
            "midi-touchup-editor.exe" if os.name == "nt" else "midi-touchup-editor",
        )
        build_cmd = "cd tools/midi_touchup_editor_rust && cargo build --release"
        QMessageBox.warning(
            self.app,
            translate("MidiTouchupController", "Touch-Up Editor Not Built"),
            translate(
                "MidiTouchupController",
                "Rust touch-up editor binary was not found.\n\nMIDI requested: {midi_path}\nExpected binary: {expected_binary}\n\nRun setup first (it can install/build Rust touch-up):\n{setup_cmd}\n\nOr build manually with:\n{build_cmd}\n\nThen retry Edit MIDI.",
            ).format(
                midi_path=midi_path,
                expected_binary=os.path.join(repo_root, expected_rel),
                setup_cmd=setup_cmd,
                build_cmd=build_cmd,
            ),
        )

    def handle_process_finished(
        self,
        process: QProcess,
        source_midi_path: str,
        exit_code: int,
    ) -> None:
        app = self.app
        try:
            stdout_text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
            stderr_text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            logging.warning(
                "[TOUCHUP-RUST] Process handle was deleted before finish handling (source=%s)",
                source_midi_path,
            )
            self.remove_process_ref(process)
            return

        self.cleanup_process(process)

        if app._is_closing:
            return

        if stderr_text.strip():
            logging.warning("[TOUCHUP-RUST-STDERR] %s", stderr_text.strip())

        result_payload: Dict[str, Any] = {}
        for line in reversed(stdout_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                result_payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

        status = str(result_payload.get("status", "")).strip().lower()
        saved_path = result_payload.get("saved_path")
        message = result_payload.get("message") or ""

        if status == "saved" and exit_code == 0:
            shown_path = saved_path if isinstance(saved_path, str) and saved_path else "(path not provided)"
            if not app._is_closing:
                self.editor_saved.emit(source_midi_path, shown_path)
                QMessageBox.information(
                    app,
                    translate("MidiTouchupController", "Touch-Up Saved"),
                    translate("MidiTouchupController", "Touch-up MIDI saved to:\n{shown_path}").format(
                        shown_path=shown_path
                    ),
                )
            return

        if status == "cancelled" and exit_code == 0:
            self.editor_cancelled.emit(source_midi_path)
            logging.info("[TOUCHUP-RUST] User cancelled editor for %s", source_midi_path)
            return

        failure_message = message or f"Rust touch-up editor exited with code {exit_code}."
        if not app._is_closing:
            self.editor_failed.emit(source_midi_path, failure_message)
            QMessageBox.critical(
                app,
                translate("MidiTouchupController", "Touch-Up Editor Error"),
                translate(
                    "MidiTouchupController",
                    "{failure_message}\n\nSource MIDI: {source_midi_path}\nStdout: {stdout_text}\nStderr: {stderr_text}",
                ).format(
                    failure_message=failure_message,
                    source_midi_path=source_midi_path,
                    stdout_text=stdout_text.strip() or "(empty)",
                    stderr_text=stderr_text.strip() or "(empty)",
                ),
            )

    def _retain_process(self, process: QProcess) -> None:
        """Own a QProcess reference until it finishes or is destroyed."""
        self.processes.append(process)

    def cleanup_process(self, process: QProcess) -> None:
        self.remove_process_ref(process)
        try:
            if process.state() != QProcess.NotRunning:
                process.terminate()
                if not process.waitForFinished(1000):
                    process.kill()
                    process.waitForFinished(500)
        except RuntimeError:
            return
        process.deleteLater()

    def remove_process_ref(self, process: QProcess) -> None:
        try:
            self.processes.remove(process)
        except ValueError:
            pass

    def shutdown_processes(self) -> None:
        if not self.processes:
            return

        for process in list(self.processes):
            self.cleanup_process(process)
