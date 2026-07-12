import json
from types import SimpleNamespace

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.workflows.conversion import ConversionWorkflow


def _conversion_state() -> AppState:
    state = AppState()
    overlay = OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=10,
        height=40,
        key_type="white",
        unlit_reference_color=(12, 12, 12),
    )
    state.overlays = [overlay]
    state.detection.exemplar_lit_colors.update(
        {
            "LW": (255, 0, 0),
            "LB": (160, 0, 0),
            "RW": (0, 120, 255),
            "RB": (0, 70, 180),
        }
    )
    return state


def _workflow(state: AppState) -> ConversionWorkflow:
    return ConversionWorkflow(
        state,
        SimpleNamespace(),
        runtime_paths=SimpleNamespace(),
    )


def test_conversion_preflight_names_enabled_higher_family_missing_color():
    state = _conversion_state()
    state.detection.exemplar_key_type_enabled["COLOR_4_B"] = True
    state.detection.exemplar_lit_colors["COLOR_4_B"] = None

    errors = _workflow(state)._validate_prerequisites()

    assert any("Color 4 Sharp / Flat" in error for error in errors)


def test_conversion_preflight_ignores_unchecked_higher_family_slot():
    state = _conversion_state()
    state.detection.exemplar_key_type_enabled["COLOR_4_B"] = False
    state.detection.exemplar_lit_colors["COLOR_4_B"] = None

    errors = _workflow(state)._validate_prerequisites()

    assert not any("Missing exemplar colors" in error for error in errors)


class _FrameLocalDetector:
    def __init__(self):
        self._matches = iter(("COLOR_3_W", "COLOR_4_W"))
        self.last_match = None

    def get_name(self):
        return "Frame-local detector"

    def detect_frame(self, *_args, **_kwargs):
        self.last_match = next(self._matches)
        return {1}

    def get_last_exemplar_match(self, key_id):
        assert key_id == 1
        return self.last_match


class _TwoFrameSession:
    width = 10
    height = 40
    total_frames = 2
    fps = 30.0

    def seek_to_frame(self, frame_idx):
        assert frame_idx == 0

    def get_frame_sequential(self):
        return True, SimpleNamespace(shape=(40, 10, 3))


def test_conversion_passes_frame_local_detector_winners_to_midi_processing(monkeypatch):
    state = _conversion_state()
    state.video.processing_start_frame = 0
    state.video.processing_end_frame = 1
    workflow = ConversionWorkflow(
        state,
        _TwoFrameSession(),
        runtime_paths=SimpleNamespace(),
    )
    received_matches = []

    def record_midi_events(
        _pressed_key_ids,
        _frame_idx,
        _active_notes,
        _midi_writer,
        _frame_bgr,
        _overlays,
        exemplar_matches=None,
    ):
        received_matches.append(exemplar_matches)
        return 0

    monkeypatch.setattr(workflow, "_process_midi_events", record_midi_events)
    midi_writer = SimpleNamespace(finalize_active_notes=lambda _final_time: None)

    success = workflow._process_frame_range(
        _FrameLocalDetector(),
        midi_writer,
        total_frames=2,
        progress_callback=lambda *_args: None,
    )

    assert success is True
    assert received_matches == [{1: "COLOR_3_W"}, {1: "COLOR_4_W"}]


def test_conversion_settings_log_serializes_active_dynamic_exemplar_slots(tmp_path):
    state = _conversion_state()
    state.video.filepath = str(tmp_path / "source.mp4")
    state.detection.exemplar_key_type_enabled["COLOR_3_W"] = True
    state.detection.exemplar_lit_colors["COLOR_3_W"] = (30, 40, 50)
    state.detection.exemplar_key_type_enabled["COLOR_4_B"] = False
    state.detection.exemplar_lit_colors["COLOR_4_B"] = (60, 70, 80)
    log_path = tmp_path / "conversion-settings.json"
    runtime_paths = SimpleNamespace(
        conversion_settings_path=lambda _video_path, _midi_path: log_path
    )
    workflow = ConversionWorkflow(
        state,
        SimpleNamespace(fps=30.0),
        runtime_paths=runtime_paths,
    )

    workflow._save_midi_settings_log(
        str(tmp_path / "output.mid"),
        frames_processed=2,
        detector_name="Test detector",
    )

    detection_parameters = json.loads(log_path.read_text())["detection_parameters"]
    expected_slots = {"LW", "LB", "RW", "RB", "COLOR_3_W", "COLOR_4_B"}
    assert set(detection_parameters["exemplar_key_type_enabled"]) == expected_slots
    assert set(detection_parameters["exemplar_lit_colors"]) == expected_slots
    assert detection_parameters["exemplar_key_type_enabled"]["COLOR_3_W"] is True
    assert detection_parameters["exemplar_key_type_enabled"]["COLOR_4_B"] is False
    assert detection_parameters["exemplar_lit_colors"]["COLOR_3_W"] == [30, 40, 50]
    assert detection_parameters["exemplar_lit_colors"]["COLOR_4_B"] == [60, 70, 80]
