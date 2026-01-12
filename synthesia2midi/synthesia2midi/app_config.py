"""
Application configuration constants and core data structures.

Defines global constants, default values, and data classes used throughout
the synthesia2midi application. Contains overlay configurations, styling defaults,
note naming conventions, and other application-wide settings.

Key Components:
- OverlayConfig: Core data structure for piano key overlays
- Note naming constants (sharp/flat variations)
- Default styling for white and black keys
- Application directories and file paths
- MIDI generation parameters
"""
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

# --- General App Config ---
APP_NAME = "synthesia2midi"
LOG_DIR = "logs"
DEBUG_FRAMES_DIR = "debug_frames"
DEFAULT_MIDI_TEMPO = 120

# --- Note Names ---
NOTE_NAMES_SHARP = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"]
NOTE_NAMES_FLAT = ["A", "B♭", "B", "C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭"]

# --- Navigation Intervals ---
FRAME_NAV_INTERVALS: List[int] = [1, 5, 10, 100]

# --- Default Overlay Geometric Styles ---
# Used if my_immortal.ini is not available or doesn't define these key types.
# Values are examples, adjust for desired default appearance.
DEFAULT_WHITE_KEY_STYLE = {"y": 500.0, "height": 150.0} # Example: Y position and Height for white keys
DEFAULT_BLACK_KEY_STYLE = {"fill": "black", "outline": "gray"}

# Idealized dimensions inspired by my_immortal_video.ini for generative wizard
IDEALIZED_WHITE_KEY_Y = 1011
IDEALIZED_WHITE_KEY_HEIGHT = 57
IDEALIZED_AVG_WHITE_KEY_WIDTH = 22

IDEALIZED_BLACK_KEY_Y = 874
IDEALIZED_BLACK_KEY_HEIGHT = 124
IDEALIZED_AVG_BLACK_KEY_WIDTH = 15 # Standard black keys are often ~2/3 width of white keys

# Factor for positioning black keys: X_black = X_ref_white + W_ref_white * THIS_FACTOR
IDEALIZED_BLACK_KEY_X_START_FACTOR = 0.60 

@dataclass
class OverlayConfig:
    """Configuration for a single key overlay."""
    key_id: int
    note_octave: int # e.g., 4 for C4
    note_name_in_octave: str # e.g. "C" for C4, "F#" for F#3
    x: float
    y: float
    width: float
    height: float
    unlit_reference_color: Optional[Tuple[int, int, int]] = None
    key_type: Optional[str] = None
    unlit_hist: Optional[np.ndarray] = None
    lit_hist:   Optional[np.ndarray] = field(default=None, repr=False) # Lit histogram (HSV)
    overlay_type: str = 'key'  # 'key' or 'spark'

    # --- runtime tracking for delta detector ---
    last_progression_ratio: float = 0.0     # previous frame's ratio
    last_is_lit:            bool  = False   # previous frame's lit flag
    prev_progression_ratio: float = 0.0     # second to last frame's ratio

    # Runtime state for delta detection (not saved to INI)
    in_forced_delta_off_state: bool = False # NEW: For strict delta-off latch logic

    def get_full_note_name(self, octave_transpose: int = 0) -> str:
        """Returns the full note name like C4, F#3.
        
        Args:
            octave_transpose: Number of octaves to transpose (-8 to +8)
        """
        transposed_octave = self.note_octave + octave_transpose
        return f"{self.note_name_in_octave}{transposed_octave}"

    def get_midi_note_number(self, octave_transpose: int = 0) -> int:
        """Calculates the MIDI note number (C4=60, A4=69, A0=21).
        
        Args:
            octave_transpose: Number of octaves to transpose (-8 to +8)
        """
        # Standard MIDI: C4 = 60, C0 = 12. Octaves are 0-indexed for this calculation relative to C.
        # NOTE_NAMES_SHARP = ["A", "A♯", "B", "C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯"]
        # Create a mapping from note name to pitch class value (0-11, C=0)
        # Support both regular # and musical ♯ symbols
        pitch_class_map = {
            "C": 0, "C#": 1, "C♯": 1, "D♭": 1,
            "D": 2, "D#": 3, "D♯": 3, "E♭": 3,
            "E": 4, "F♭": 4, # E#, F♭ are enharmonically E, F
            "F": 5, "E#": 5, "E♯": 5, "F#": 6, "F♯": 6, "G♭": 6,
            "G": 7, "G#": 8, "G♯": 8, "A♭": 8,
            "A": 9, "A#": 10, "A♯": 10, "B♭": 10,
            "B": 11, "C♭": 11 # B#, C♭ are enharmonically B, C
        }
        
        note_name_clean = self.note_name_in_octave.upper()
        if note_name_clean not in pitch_class_map:
            # Fallback for any unexpected note names, though NOTE_NAMES_SHARP should cover it
            # This could happen if note_name_in_octave was, e.g., just "A" from NOTE_NAMES_SHARP
            # and not a flat name from NOTE_NAMES_FLAT that might also be used.
            # For safety, let's try a direct index from NOTE_NAMES_SHARP if primary map fails
            # to map to a C-based index.
            # C = 0, C# = 1, ... B = 11
            c_based_sharp_notes = NOTE_NAMES_SHARP[3:] + NOTE_NAMES_SHARP[:3] # C, C#, D, ..., A, A#, B
            try:
                pitch_class = c_based_sharp_notes.index(note_name_clean)
            except ValueError:
                logging.error(f"Cannot determine pitch class for note: {self.note_name_in_octave}")
                return 0 # Return a default or raise error
        else:
            pitch_class = pitch_class_map[note_name_clean]

        # self.note_octave is assumed to be the standard octave number (e.g., C4 is octave 4)
        # MIDI note number = (octave + 1) * 12 + pitch_class (if C0 is octave 0)
        # Or, more directly: (octave * 12) + pitch_class + 12 (where C is pitch_class 0)
        # C4 (octave 4, pitch_class 0) -> (4 * 12) + 0 + 12 = 60
        # A0 (octave 0, pitch_class 9) -> (0 * 12) + 9 + 12 = 21
        # A2 (octave 2, pitch_class 9) -> (2 * 12) + 9 + 12 = 24 + 9 + 12 = 45
        # E3 (octave 3, pitch_class 4) -> (3 * 12) + 4 + 12 = 36 + 4 + 12 = 52
        # A3 (octave 3, pitch_class 9) -> (3 * 12) + 9 + 12 = 36 + 9 + 12 = 57
        # C#4 (octave 4, pitch_class 1) -> (4*12) + 1 + 12 = 48 + 1 + 12 = 61
        # E4 (octave 4, pitch_class 4) -> (4*12) + 4 + 12 = 48 + 4 + 12 = 64
        # A4 (octave 4, pitch_class 9) -> (4*12) + 9 + 12 = 48 + 9 + 12 = 69

        # Apply octave transpose
        transposed_octave = self.note_octave + octave_transpose
        midi_note = (transposed_octave * 12) + pitch_class + 12
        
        # DEBUG LOGGING
        logging.debug(f"[MIDI-CALC] key_id={self.key_id}: {self.note_name_in_octave}{self.note_octave} -> pitch_class={pitch_class}, octave={self.note_octave}, transpose={octave_transpose}, midi_note={midi_note}")
        
        # Validate MIDI note range (0-127)
        if midi_note < 0:
            logging.warning(f"MIDI note {midi_note} for {self.get_full_note_name(octave_transpose)} is below 0, clamping to 0")
            midi_note = 0
        elif midi_note > 127:
            logging.warning(f"MIDI note {midi_note} for {self.get_full_note_name(octave_transpose)} is above 127, clamping to 127")
            midi_note = 127
        
        return midi_note


# --- Logging Configuration ---
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_DEBUG = logging.DEBUG

# --- Debug Frame Saving Config ---
DEBUG_FRAME_SAVE_INTERVAL = 100 # Save every 100th processed frame in debug mode