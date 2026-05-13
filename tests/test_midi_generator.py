from pathlib import Path

from synthesia2midi.midi_generator import MidiWriter


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
