"""
MIDI file generation from video detection results.

Converts detected piano key press events from video analysis into MIDI files
using the midiutil library. Handles note timing, velocity, and proper MIDI
formatting to create playable musical files from visual piano performance data.

Features:
- Note-on/note-off event management
- Configurable timing and velocity
- Multi-track MIDI support
- Duplicate note removal
- Comprehensive logging of conversion process
"""
import os
import logging
from midiutil.MidiFile import MIDIFile # type: ignore
from typing import List, Dict, Tuple

class MidiWriter:
    """Handles the creation and saving of MIDI data."""

    def __init__(self, num_tracks: int = 1, midi_file_format: int = 1, remove_duplicates: bool = True):
        """
        Initializes the MIDIFile object.
        Args:
            num_tracks: Number of tracks for the MIDI file.
            midi_file_format: MIDI file format (0, 1, or 2).
                               Format 1 is generally preferred for multi-track.
            remove_duplicates: Whether midiutil should remove duplicate notes.
        """
        self.logger = logging.getLogger(f"{__name__}.MidiWriter")
        self.debug = 0
        self.notes_buffer: List[Dict] = [] # Buffer for notes before adding to MIDIFile
        self.miditrackname: str = 'synthesia2midi Output'
        self.tempo: int = 120 # Default tempo
        
        # Track active notes for note-on/note-off workflow
        # Format: {(track, channel, pitch): start_time}
        self.active_notes: Dict[Tuple[int, int, int], float] = {}
        
        # MIDIFile parameters:
        # numTracks, removeDuplicates, deinterleave, adjust_origin, file_format
        self.mf = MIDIFile(numTracks=num_tracks,
                           removeDuplicates=remove_duplicates,
                           deinterleave=False, # Typically False for format 1
                           adjust_origin=False, # Keep absolute time
                           file_format=midi_file_format)

    def set_track_name(self, track: int, time: float, name: str) -> None:
        """Sets the name for a given track."""
        self.miditrackname = name # Store for reference, though MIDIFile handles internal name
        self.mf.addTrackName(track, time, name)

    def set_tempo(self, track: int, time: float, tempo: int) -> None:
        """Sets the tempo for a given track."""
        self.tempo = tempo
        self.mf.addTempo(track, time, tempo)

    def add_program_change(self, track: int, channel: int, time: float, program: int) -> None:
        """Adds a program change event (instrument change)."""
        self.mf.addProgramChange(track, channel, time, program)

    def add_note_to_buffer(self, track: int, channel: int, pitch: int, 
                           start_time_beats: float, duration_beats: float, volume: int = 100) -> None:
        """
        Adds a note to an internal buffer. Notes from the buffer are written to the
        MIDIFile object when save_to_disk is called.
        Args:
            track: The track number.
            channel: The MIDI channel (0-15).
            pitch: MIDI note number (0-127).
            start_time_beats: Note start time in beats.
            duration_beats: Note duration in beats.
            volume: Note velocity (0-127).
        """
        self.notes_buffer.append({
            'track': track, 
            'channel': channel, 
            'pitch': pitch, 
            'start_time': start_time_beats, 
            'duration': duration_beats, 
            'volume': volume
        })

    def _commit_buffer_to_midifile(self) -> None:
        """Transfers all notes from the internal buffer to the MIDIFile object."""
        for note_params in self.notes_buffer:
            self.mf.addNote(
                note_params['track'],
                note_params['channel'],
                note_params['pitch'],
                note_params['start_time'],
                note_params['duration'],
                note_params['volume']
            )
        self.notes_buffer.clear() # Clear buffer after committing

    def add_note_on(self, track: int, channel: int, time: float, pitch: int, velocity: int = 80) -> None:
        """
        Adds a note-on event. Tracks the start time for later note-off processing.
        Args:
            track: The track number.
            channel: The MIDI channel (0-15).
            time: Time in beats when the note starts.
            pitch: MIDI note number (0-127).
            velocity: Note velocity (0-127).
        """
        note_key = (track, channel, pitch)
        # If this note is already active, end it first
        if note_key in self.active_notes:
            start_time = self.active_notes[note_key]
            duration = max(0.01, time - start_time)  # Minimum duration to avoid zero-length notes
            self.mf.addNote(track, channel, pitch, start_time, duration, velocity)
        
        # Start tracking this note
        self.active_notes[note_key] = time
    
    def add_note_off(self, track: int, channel: int, time: float, pitch: int, velocity: int = 80) -> None:
        """
        Adds a note-off event by completing the note with proper duration.
        Args:
            track: The track number.
            channel: The MIDI channel (0-15).
            time: Time in beats when the note ends.
            pitch: MIDI note number (0-127).
            velocity: Note velocity (0-127).
        """
        note_key = (track, channel, pitch)
        if note_key in self.active_notes:
            start_time = self.active_notes.pop(note_key)
            duration = max(0.01, time - start_time)  # Minimum duration to avoid zero-length notes
            self.mf.addNote(track, channel, pitch, start_time, duration, velocity)
    
    def save_file(self, filename: str) -> bool:
        """
        Alias for save_to_disk that returns only success status.
        Args:
            filename: The path to save the MIDI file.
        Returns:
            True if successful, False otherwise.
        """
        self.logger.warning(f"[MIDI-SAVE-FILE] Called save_file for: {filename}")
        success, message = self.save_to_disk(filename)
        self.logger.warning(f"[MIDI-SAVE-FILE] Result: success={success}, message={message}")
        return success

    def finalize_active_notes(self, final_time: float = None) -> None:
        """
        Finalizes any remaining active notes with a default duration.
        Should be called before saving to ensure no notes are left hanging.
        Args:
            final_time: The final time in beats. If None, uses current active note time + 0.5 beats.
        """
        self.logger.info(f"[FINALIZE-NOTES] Starting finalization of {len(self.active_notes)} active notes")
        for note_key, start_time in list(self.active_notes.items()):
            track, channel, pitch = note_key
            if final_time is not None:
                duration = max(0.01, final_time - start_time)
            else:
                duration = 0.5  # Default duration of half a beat
            self.mf.addNote(track, channel, pitch, start_time, duration, 80)
        
        self.logger.info(f"[FINALIZE-NOTES] Cleared {len(self.active_notes)} active notes")
        self.active_notes.clear()

    def save_to_disk(self, filename: str) -> Tuple[bool, str]:
        """
        Commits buffered notes and writes the MIDI data to a file.
        Args:
            filename: The path to save the MIDI file.
        Returns:
            A tuple (success, message).
        """
        self.logger.warning("[SAVE-TO-DISK-START] Starting save_to_disk operation")
        
        # Finalize any remaining active notes
        self.logger.info("[SAVE-TO-DISK] Finalizing active notes...")
        self.finalize_active_notes()
        
        if not self.notes_buffer:
            # Allow saving an empty MIDI file if tracks/tempo were set up
            # return False, 'No notes to save.'
            self.logger.info(f"[SAVE-TO-DISK] Notes buffer is empty, but continuing with save")
            pass
        else:
            self.logger.info(f"[SAVE-TO-DISK] Notes buffer contains {len(self.notes_buffer)} notes")

        self.logger.info("[SAVE-TO-DISK] Committing buffer to MIDI file...")
        self._commit_buffer_to_midifile()
        
        try:
            self.logger.warning(f"[SAVE-TO-DISK-WRITE] Creating directory and writing file: {filename}")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as outf:
                self.logger.warning("[SAVE-TO-DISK-WRITE] Calling midiutil writeFile...")
                self.mf.writeFile(outf)
                self.logger.warning("[SAVE-TO-DISK-WRITE] writeFile completed successfully")
            self.logger.warning("[SAVE-TO-DISK-SUCCESS] File saved successfully")
            return True, f'Saved to disk: {filename}'
        except Exception as e:
            self.logger.error(f"[SAVE-TO-DISK-ERROR] Failed to save: {e}", exc_info=True)
            return False, f"Can't save to disk: {filename}. Error: {e}"