from types import SimpleNamespace

from synthesia2midi.gui.midi_conversion_controller import MidiConversionController
from synthesia2midi.workflows.midi_export import MidiExportService


class FakeControlPanel:
    def __init__(self):
        self.results = []

    def set_conversion_result(self, success, message):
        self.results.append((success, message))


class FakeTouchupController:
    def __init__(self):
        self.completed_paths = []

    def show_conversion_complete_dialog(self, midi_path):
        self.completed_paths.append(midi_path)


class FakeWorkflow:
    def __init__(self, success=True):
        self.success = success
        self.paths = []

    def convert_to_midi(self, output_path):
        self.paths.append(output_path)
        return self.success


def _fake_app(video_path, workflow):
    return SimpleNamespace(
        app_state=SimpleNamespace(video=SimpleNamespace(filepath=str(video_path), original_video_path=None)),
        conversion_workflow=workflow,
        control_panel=FakeControlPanel(),
        midi_touchup_controller=FakeTouchupController(),
    )


def test_start_conversion_without_workflow_reports_missing_video(monkeypatch):
    app = _fake_app("/tmp/video.mp4", None)
    info_calls = []
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_conversion_controller.QMessageBox.information",
        lambda *args: info_calls.append(args),
    )

    MidiConversionController(app).start_conversion_process()

    assert app.control_panel.results == [(False, "Please open a video file first.")]
    assert info_calls


def test_start_conversion_uses_original_video_path_and_prompts_touchup(monkeypatch, tmp_path):
    original_video = tmp_path / "song.mp4"
    workflow = FakeWorkflow(success=True)
    app = _fake_app(tmp_path / "frames", workflow)
    app.app_state.video.original_video_path = str(original_video)

    class FixedDateTime:
        @classmethod
        def now(cls):
            class FixedNow:
                def strftime(self, _fmt):
                    return "20260512_101500"
            return FixedNow()

    monkeypatch.setattr("synthesia2midi.workflows.midi_export.datetime.datetime", FixedDateTime)

    MidiConversionController(app).start_conversion_process()

    expected = tmp_path / "Completed MIDI Files" / "song_20260512_101500.mid"
    assert workflow.paths == [str(expected)]
    assert app.control_panel.results == [(True, f"MIDI file saved to:\n{expected}")]
    assert app.midi_touchup_controller.completed_paths == [str(expected)]


def test_start_conversion_failure_reports_failure_without_touchup(monkeypatch, tmp_path):
    critical_calls = []
    monkeypatch.setattr(
        "synthesia2midi.gui.midi_conversion_controller.QMessageBox.critical",
        lambda *args: critical_calls.append(args),
    )
    workflow = FakeWorkflow(success=False)
    app = _fake_app(tmp_path / "song.mp4", workflow)

    MidiConversionController(app).start_conversion_process()

    assert app.control_panel.results == [(False, "MIDI conversion failed. Check logs for details.")]
    assert critical_calls
    assert app.midi_touchup_controller.completed_paths == []


def test_midi_export_service_builds_default_path_from_original_video(monkeypatch, tmp_path):
    original_video = tmp_path / "song.mp4"
    workflow = FakeWorkflow(success=True)
    app_state = SimpleNamespace(video=SimpleNamespace(filepath=str(tmp_path / "frames"), original_video_path=str(original_video)))

    class FixedDateTime:
        @classmethod
        def now(cls):
            class FixedNow:
                def strftime(self, _fmt):
                    return "20260512_101500"
            return FixedNow()

    monkeypatch.setattr("synthesia2midi.workflows.midi_export.datetime.datetime", FixedDateTime)

    result = MidiExportService(app_state, workflow).export_to_default_path()

    expected = tmp_path / "Completed MIDI Files" / "song_20260512_101500.mid"
    assert result.success is True
    assert result.output_path == str(expected)
    assert result.message == f"MIDI file saved to:\n{expected}"
    assert workflow.paths == [str(expected)]
