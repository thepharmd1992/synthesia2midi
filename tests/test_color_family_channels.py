from types import SimpleNamespace

import pytest

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.workflows import conversion
from synthesia2midi.workflows.conversion import ConversionWorkflow


@pytest.mark.parametrize(
    ("slot", "expected_channel"),
    [
        ("LW", 0),
        ("LB", 0),
        ("RW", 1),
        ("RB", 1),
        ("COLOR_3_W", 2),
        ("COLOR_3_B", 2),
        ("COLOR_4_W", 3),
        ("COLOR_4_B", 3),
    ],
)
def test_all_exemplar_slots_map_to_their_family_channel(slot, expected_channel):
    assert conversion._midi_channel_for_exemplar(slot) == expected_channel


def test_missing_exemplar_identity_maps_to_default_channel():
    assert conversion._midi_channel_for_exemplar(None) == 0


class RecordingMidiWriter:
    def __init__(self):
        self.note_ons = []

    def add_note_on(self, track, channel, time, pitch, velocity):
        self.note_ons.append((track, channel, time, pitch, velocity))


def _workflow_with_overlays(count=4):
    state = AppState()
    state.detection.hand_assignment_enabled = True
    state.overlays = [
        OverlayConfig(
            key_id=key_id,
            note_octave=4,
            note_name_in_octave=note_name,
            x=key_id * 10,
            y=0,
            width=8,
            height=20,
            key_type="white",
            unlit_reference_color=(12, 12, 12),
        )
        for key_id, note_name in enumerate(("C", "D", "E", "F")[:count], start=1)
    ]
    workflow = ConversionWorkflow(
        state,
        SimpleNamespace(fps=30.0),
        runtime_paths=SimpleNamespace(),
    )
    return workflow, state.overlays


def test_simultaneous_family_winners_start_notes_on_four_channels(monkeypatch):
    workflow, overlays = _workflow_with_overlays()
    midi_writer = RecordingMidiWriter()
    monkeypatch.setattr(
        workflow,
        "_determine_hand_channel",
        lambda *_args: pytest.fail("known detector identities must not use color fallback"),
    )

    notes_created = workflow._process_midi_events(
        [1, 2, 3, 4],
        frame_idx=0,
        active_notes={},
        midi_writer=midi_writer,
        frame_bgr=None,
        overlays=overlays,
        exemplar_matches={
            1: "LW",
            2: "RB",
            3: "COLOR_3_W",
            4: "COLOR_4_B",
        },
    )

    assert notes_created == 4
    assert [event[1] for event in midi_writer.note_ons] == [0, 1, 2, 3]


def test_detector_without_identity_uses_nearest_color_fallback(monkeypatch):
    workflow, overlays = _workflow_with_overlays(count=1)
    midi_writer = RecordingMidiWriter()
    fallback_calls = []

    def fallback(overlay, frame_bgr):
        fallback_calls.append((overlay.key_id, frame_bgr))
        return 2

    monkeypatch.setattr(workflow, "_determine_hand_channel", fallback)

    workflow._process_midi_events(
        {1},
        frame_idx=0,
        active_notes={},
        midi_writer=midi_writer,
        frame_bgr="frame",
        overlays=overlays,
        exemplar_matches={1: None},
    )

    assert fallback_calls == [(1, "frame")]
    assert [event[1] for event in midi_writer.note_ons] == [2]


def test_known_identity_preserves_single_channel_mode_when_assignment_is_disabled(
    monkeypatch,
):
    workflow, overlays = _workflow_with_overlays(count=1)
    workflow.app_state.detection.hand_assignment_enabled = False
    midi_writer = RecordingMidiWriter()
    monkeypatch.setattr(
        workflow,
        "_determine_hand_channel",
        lambda *_args: pytest.fail("known detector identities must not be recomputed"),
    )

    workflow._process_midi_events(
        {1},
        frame_idx=0,
        active_notes={},
        midi_writer=midi_writer,
        frame_bgr=None,
        overlays=overlays,
        exemplar_matches={1: "COLOR_4_W"},
    )

    assert [event[1] for event in midi_writer.note_ons] == [0]
