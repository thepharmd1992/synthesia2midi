"""
MIDI file reader for analysis and comparison
"""
from dataclasses import dataclass
from typing import List

@dataclass
class Note:
    """Represents a MIDI note"""
    pitch: int
    start: float
    end: float
    velocity: int = 100
    
    @property
    def duration(self):
        return self.end - self.start

class MidiReader:
    """Reads and parses MIDI files"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.notes = []
        self._parse_file()
        
    def _parse_file(self):
        """Parse MIDI file using mido for accurate reading"""
        try:
            import mido
        except ImportError:
            # Fallback to basic parsing if mido not available
            self._parse_basic()
            return
            
        mid = mido.MidiFile(self.filepath)
        
        # Track time and active notes
        current_time = 0
        active_notes = {}  # pitch -> (start_time, velocity)
        
        for track in mid.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Note on
                    active_notes[msg.note] = (current_time, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # Note off
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note]
                        self.notes.append(Note(
                            pitch=msg.note,
                            start=start_time,
                            end=current_time,
                            velocity=velocity
                        ))
                        del active_notes[msg.note]
        
        # Sort notes by start time
        self.notes.sort(key=lambda n: n.start)
        
    def _parse_basic(self):
        """Basic MIDI parsing fallback"""
        # This is a simplified parser - in production would need full MIDI parsing
        # For now, we'll rely on mido being installed
        pass
        
    def get_notes(self) -> List[Note]:
        """Get all notes from the MIDI file"""
        return self.notes
        
    def get_notes_in_range(self, start_time: float, end_time: float) -> List[Note]:
        """Get notes within a specific time range"""
        return [n for n in self.notes if n.start >= start_time and n.start < end_time]
