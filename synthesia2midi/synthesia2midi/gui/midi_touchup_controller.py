"""Rust MIDI touch-up editor integration for the Qt app."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QFileDialog, QMessageBox


class MidiTouchupController:
    """Owns the Rust MIDI touch-up editor process lifecycle."""

    def __init__(self, app):
        self.app = app
        self.processes: list[QProcess] = []

    def show_conversion_complete_dialog(self, midi_output_path: str) -> None:
        app = self.app
        msg_box = QMessageBox(app)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Conversion Complete")
        msg_box.setText(f"MIDI file saved to:\n{midi_output_path}")
        msg_box.setInformativeText("You can open the Touch-Up Editor now, or finish.")
        open_btn = msg_box.addButton("Open Touch-Up Editor", QMessageBox.ActionRole)
        done_btn = msg_box.addButton("Done", QMessageBox.AcceptRole)
        msg_box.setDefaultButton(open_btn)
        msg_box.exec()

        if msg_box.clickedButton() is open_btn:
            self.open_editor(midi_output_path)
        elif msg_box.clickedButton() is done_btn:
            return

    def open_from_picker(self) -> None:
        repo_root = self._repo_root()
        videos_dir = os.path.join(repo_root, "videos")
        start_dir = videos_dir if os.path.isdir(videos_dir) else os.path.expanduser("~")

        midi_path, _ = QFileDialog.getOpenFileName(
            self.app,
            "Open MIDI for Touch-Up",
            start_dir,
            "MIDI Files (*.mid *.midi)",
        )
        if not midi_path:
            return
        self.open_editor(midi_path)

    def open_editor(self, midi_path: str) -> None:
        app = self.app
        if not os.path.exists(midi_path):
            QMessageBox.warning(app, "Touch-Up Editor", f"MIDI file not found:\n{midi_path}")
            return

        binary_path = self.resolve_binary_path()
        if binary_path is None:
            self.show_setup_dialog(midi_path)
            return

        process = QProcess(app)
        process.setProgram(binary_path)
        process.setArguments(["--midi", midi_path, "--result-json", "--theme", "neothesia"])
        process.setProcessChannelMode(QProcess.SeparateChannels)
        self.processes.append(process)
        app._midi_touchup_processes = self.processes

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
                QMessageBox.critical(
                    app,
                    "Touch-Up Editor Launch Failed",
                    (
                        "Failed to start Rust touch-up editor.\n\n"
                        f"Binary: {binary_path}\n"
                        f"Error: {error_msg}"
                    ),
                )
            return

    def resolve_binary_path(self) -> Optional[str]:
        repo_root = self._repo_root()
        binary_name = "midi-touchup-editor.exe" if os.name == "nt" else "midi-touchup-editor"
        primary = os.path.join(
            repo_root, "tools", "midi_touchup_editor_rust", "target", "release", binary_name
        )
        alternate = os.path.join(
            repo_root,
            "tools",
            "midi_touchup_editor_rust",
            "target",
            "release",
            "midi_touchup_editor_rust.exe" if os.name == "nt" else "midi_touchup_editor_rust",
        )

        for candidate in (primary, alternate):
            if os.path.isfile(candidate):
                if os.name == "nt" or os.access(candidate, os.X_OK):
                    return candidate
        return None

    def show_setup_dialog(self, midi_path: str) -> None:
        repo_root = self._repo_root()
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
            "Touch-Up Editor Not Built",
            (
                "Rust touch-up editor binary was not found.\n\n"
                f"MIDI requested: {midi_path}\n"
                f"Expected binary: {os.path.join(repo_root, expected_rel)}\n\n"
                "Run setup first (it can install/build Rust touch-up):\n"
                f"{setup_cmd}\n\n"
                "Or build manually with:\n"
                f"{build_cmd}\n\n"
                "Then retry Edit MIDI."
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
                QMessageBox.information(
                    app,
                    "Touch-Up Saved",
                    f"Touch-up MIDI saved to:\n{shown_path}",
                )
            return

        if status == "cancelled" and exit_code == 0:
            logging.info("[TOUCHUP-RUST] User cancelled editor for %s", source_midi_path)
            return

        failure_message = message or f"Rust touch-up editor exited with code {exit_code}."
        if not app._is_closing:
            QMessageBox.critical(
                app,
                "Touch-Up Editor Error",
                (
                    f"{failure_message}\n\n"
                    f"Source MIDI: {source_midi_path}\n"
                    f"Stdout: {stdout_text.strip() or '(empty)'}\n"
                    f"Stderr: {stderr_text.strip() or '(empty)'}"
                ),
            )

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
        self.app._midi_touchup_processes = self.processes

    def shutdown_processes(self) -> None:
        if not self.processes:
            return

        for process in list(self.processes):
            self.cleanup_process(process)

    @staticmethod
    def _repo_root() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
