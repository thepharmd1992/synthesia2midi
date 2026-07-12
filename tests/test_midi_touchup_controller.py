from types import SimpleNamespace

from PySide6.QtCore import QObject

from synthesia2midi.gui.midi_touchup_controller import MidiTouchupController
from synthesia2midi.runtime_paths import RuntimePaths


class RecordingSignal:
    def __init__(self):
        self.connected = []

    def connect(self, slot):
        self.connected.append(slot)


class FakeProcess:
    SeparateChannels = object()
    NotRunning = object()

    instances = []

    def __init__(self, parent=None):
        self.parent = parent
        self.program = None
        self.arguments = None
        self.channel_mode = None
        self.destroyed = RecordingSignal()
        self.finished = RecordingSignal()
        self.started = False
        self.deleted = False
        FakeProcess.instances.append(self)

    def setProgram(self, program):
        self.program = program

    def setArguments(self, arguments):
        self.arguments = list(arguments)

    def setProcessChannelMode(self, mode):
        self.channel_mode = mode

    def start(self):
        self.started = True

    def waitForStarted(self, timeout_ms):
        self.wait_timeout_ms = timeout_ms
        return True

    def state(self):
        return self.NotRunning

    def deleteLater(self):
        self.deleted = True


class FinishedProcess:
    def __init__(self, stdout_text, stderr_text=""):
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text

    def readAllStandardOutput(self):
        return self.stdout_text.encode("utf-8")

    def readAllStandardError(self):
        return self.stderr_text.encode("utf-8")


def _fake_app():
    return SimpleNamespace(_is_closing=False)


def test_controller_is_qobject_and_exposes_lifecycle_signals():
    controller = MidiTouchupController(_fake_app())

    assert isinstance(controller, QObject)
    for signal_name in (
        "editor_started",
        "editor_saved",
        "editor_cancelled",
        "editor_failed",
        "setup_required",
    ):
        assert hasattr(controller, signal_name)


def test_open_editor_retains_process_and_emits_started_signal(monkeypatch, tmp_path):
    FakeProcess.instances.clear()
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")
    app = _fake_app()
    controller = MidiTouchupController(app)
    started = []

    controller.editor_started.connect(lambda source, binary: started.append((source, binary)))
    monkeypatch.setattr("synthesia2midi.gui.midi_touchup_controller.QProcess", FakeProcess)
    monkeypatch.setattr(controller, "resolve_binary_path", lambda: "/bin/midi-touchup-editor")

    controller.open_editor(str(midi_path))

    process = FakeProcess.instances[0]
    assert process.program == "/bin/midi-touchup-editor"
    assert process.arguments == ["--midi", str(midi_path), "--result-json", "--theme", "neothesia"]
    assert process.channel_mode is FakeProcess.SeparateChannels
    assert process.started is True
    assert process.wait_timeout_ms == 2000
    assert controller.processes == [process]
    assert started == [(str(midi_path), "/bin/midi-touchup-editor")]


def test_handle_process_finished_uses_last_valid_stdout_json_and_emits_saved(monkeypatch, tmp_path):
    app = _fake_app()
    controller = MidiTouchupController(app)
    process = FinishedProcess(
        "log line\n"
        '{"status":"error","message":"ignored older result"}\n'
        '{"status":"saved","source_path":"song.mid","saved_path":"song_touchup.mid","message":"ok"}\n'
    )
    info_calls = []
    saved = []
    cleaned = []

    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.QMessageBox.information",
        lambda *args: info_calls.append(args),
    )
    monkeypatch.setattr(controller, "cleanup_process", lambda proc: cleaned.append(proc))
    controller.editor_saved.connect(lambda source, saved_path: saved.append((source, saved_path)))

    controller.handle_process_finished(process, str(tmp_path / "song.mid"), 0)

    assert cleaned == [process]
    assert len(info_calls) == 1
    assert "song_touchup.mid" in info_calls[0][2]
    assert saved == [(str(tmp_path / "song.mid"), "song_touchup.mid")]


def test_handle_process_finished_emits_failure_for_error_result(monkeypatch, tmp_path):
    app = _fake_app()
    controller = MidiTouchupController(app)
    process = FinishedProcess(
        'diagnostic output\n{"status":"error","message":"load failed"}\n',
        "stderr details " * 2000,
    )
    failures = []

    class FakeMessageBox:
        Critical = object()

        def __init__(self, parent=None):
            self.parent = parent
            FakeMessageBox.latest = self

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.window_title = title

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def setDetailedText(self, text):
            self.detailed_text = text

        def exec(self):
            self.executed = True

    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.QMessageBox",
        FakeMessageBox,
    )
    monkeypatch.setattr(controller, "cleanup_process", lambda proc: None)
    controller.editor_failed.connect(lambda source, message: failures.append((source, message)))

    controller.handle_process_finished(process, str(tmp_path / "song.mid"), 1)

    message_box = FakeMessageBox.latest
    assert message_box.executed is True
    assert message_box.text == "load failed"
    assert str(tmp_path / "song.mid") in message_box.informative_text
    assert "diagnostic output" in message_box.detailed_text
    assert "stderr details" in message_box.detailed_text
    assert len(message_box.detailed_text) <= 12050
    assert failures == [(str(tmp_path / "song.mid"), "load failed")]


def test_conversion_complete_dialog_has_editor_and_show_folder(monkeypatch, tmp_path):
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")
    app = _fake_app()
    controller = MidiTouchupController(app)
    shown_actions = []

    class FakeMessageBox:
        Information = object()
        ActionRole = object()
        AcceptRole = object()

        def __init__(self, parent=None):
            self.parent = parent
            self.buttons = []
            self._clicked = None
            self.button_by_text = {}
            FakeMessageBox.latest = self

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.window_title = title

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def addButton(self, text, role):
            self.buttons.append((text, role))
            button = object()
            self.button_by_text[text] = button
            if text == "Show MIDI in Folder":
                self._clicked = button
            return button

        def setDefaultButton(self, button):
            self.default_button = button

        def exec(self):
            shown_actions.extend(text for text, _role in self.buttons)

        def clickedButton(self):
            return self._clicked

    monkeypatch.setattr("synthesia2midi.gui.midi_touchup_controller.QMessageBox", FakeMessageBox)
    monkeypatch.setattr(controller, "_show_midi_in_folder", lambda path: shown_actions.append(("revealed", path)))

    controller.show_conversion_complete_dialog(str(midi_path))

    assert "Open Touch-Up Editor" in shown_actions
    assert "Show MIDI in Folder" in shown_actions
    assert "Done" not in shown_actions
    assert FakeMessageBox.latest.default_button is FakeMessageBox.latest.button_by_text["Show MIDI in Folder"]
    assert ("revealed", str(midi_path)) in shown_actions


def test_show_midi_in_folder_uses_macos_reveal(monkeypatch, tmp_path):
    midi_path = tmp_path / "song.mid"
    calls = []
    controller = MidiTouchupController(_fake_app())

    monkeypatch.setattr("synthesia2midi.gui.midi_touchup_controller.sys.platform", "darwin")
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.subprocess.run",
        lambda args, check=False: calls.append((args, check)),
    )

    controller._show_midi_in_folder(str(midi_path))

    assert calls == [(["open", "-R", str(midi_path)], False)]


def test_resolve_binary_path_uses_runtime_paths(monkeypatch, tmp_path):
    app = _fake_app()
    controller = MidiTouchupController(app)
    binary_path = tmp_path / "bundle" / "bin" / "midi-touchup-editor"
    binary_path.parent.mkdir(parents=True)
    binary_path.write_text("", encoding="utf-8")
    binary_path.chmod(0o755)

    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.detect_runtime_paths",
        lambda: RuntimePaths(
            frozen=True,
            app_root=tmp_path / "bundle",
            repo_root=tmp_path / "repo",
            home_dir=tmp_path / "home",
            platform_name="darwin",
        ),
    )

    assert controller.resolve_binary_path() == str(binary_path)


def test_show_setup_dialog_uses_packaged_message_when_frozen(monkeypatch, tmp_path):
    app = _fake_app()
    controller = MidiTouchupController(app)
    warnings = []

    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.detect_runtime_paths",
        lambda: RuntimePaths(
            frozen=True,
            app_root=tmp_path / "bundle",
            repo_root=tmp_path / "repo",
            home_dir=tmp_path / "home",
            platform_name="darwin",
        ),
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_touchup_controller.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    controller.show_setup_dialog(str(tmp_path / "song.mid"))

    assert len(warnings) == 1
    assert "Bundled Rust touch-up editor files were not found." in warnings[0][2]
