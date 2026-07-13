from pathlib import Path

import pytest

from synthesia2midi.midi_generator import (
    COLOR_MAP_META_PREFIX,
    MidiWriter,
    serialize_channel_color_map,
)


def test_channel_color_metadata_is_deterministic_and_compact():
    payload = serialize_channel_color_map(
        {
            1: {"sharp_flat": (12, 34, 56), "natural": (90, 120, 150)},
            0: {"natural": (1, 2, 3)},
        }
    )

    assert payload == (
        COLOR_MAP_META_PREFIX
        + '{"channels":{"0":{"natural":[1,2,3]},'
        '"1":{"natural":[90,120,150],"sharp_flat":[12,34,56]}}}'
    )
    assert len(payload.encode("utf-8")) <= 4096


@pytest.mark.parametrize(
    "channel_colors",
    [
        {},
        {-1: {"natural": (1, 2, 3)}},
        {16: {"natural": (1, 2, 3)}},
        {True: {"natural": (1, 2, 3)}},
        {0: {"natural": (1, 2, 3)}, "1": {"natural": (4, 5, 6)}},
        {0: {}},
        {0: {"natural": (1, 2)}},
        {0: {"natural": (1, 2, 999)}},
        {0: {"unknown": (1, 2, 3)}},
    ],
)
def test_channel_color_metadata_rejects_invalid_values(channel_colors):
    with pytest.raises(ValueError):
        serialize_channel_color_map(channel_colors)


def test_midi_writer_adds_one_channel_color_text_event():
    writer = MidiWriter(midi_file_format=1)

    writer.add_channel_color_map({0: {"natural": (1, 2, 3)}})

    text_events = [
        event
        for track in writer.mf.tracks
        for event in track.eventList
        if type(event).__name__ == "Text"
    ]
    assert len(text_events) == 1
    assert text_events[0].text.decode("ascii").startswith(COLOR_MAP_META_PREFIX)


def test_midi_writer_saves_when_filename_has_no_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = MidiWriter(midi_file_format=1)
    writer.set_track_name(0, 0, "test")
    writer.set_tempo(0, 0, 120)
    writer.add_note_on(0, 0, 0.0, 60)
    writer.add_note_off(0, 0, 1.0, 60)

    success, message = writer.save_to_disk("out.mid")

    assert success, message
    assert Path("out.mid").is_file()


def test_midi_writer_saves_when_directory_is_present(tmp_path):
    output = tmp_path / "nested" / "out.mid"
    writer = MidiWriter(midi_file_format=1)
    writer.set_track_name(0, 0, "test")
    writer.set_tempo(0, 0, 120)
    writer.add_note_on(0, 0, 0.0, 60)
    writer.add_note_off(0, 0, 1.0, 60)

    success, message = writer.save_to_disk(str(output))

    assert success, message
    assert output.is_file()
