from synthesia2midi.app_config import OverlayConfig


def overlay(note_name: str, octave: int) -> OverlayConfig:
    return OverlayConfig(
        key_id=1,
        note_octave=octave,
        note_name_in_octave=note_name,
        x=0,
        y=0,
        width=10,
        height=10,
    )


def test_overlay_config_midi_note_mapping_standard_notes():
    assert overlay("C", 4).get_midi_note_number() == 60
    assert overlay("A", 4).get_midi_note_number() == 69
    assert overlay("A", 0).get_midi_note_number() == 21


def test_overlay_config_midi_note_mapping_enharmonics_and_transpose():
    assert overlay("C#", 4).get_midi_note_number() == 61
    assert overlay("D♭", 4).get_midi_note_number() == 61
    assert overlay("F♯", 3).get_midi_note_number(octave_transpose=1) == 66


def test_overlay_config_midi_note_clamps_to_valid_range():
    assert overlay("C", -5).get_midi_note_number() == 0
    assert overlay("G", 10).get_midi_note_number(octave_transpose=8) == 127
