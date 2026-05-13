"""Note-string parsing helpers for detector assignment code."""


class NoteParsingMixin:
    def _extract_note_name(self, full_note):
        """Extract note class (e.g. C, F#, B) from note+octave string."""
        if not full_note:
            return ""
        idx = 0
        while idx < len(full_note) and not (full_note[idx].isdigit() or full_note[idx] == "-"):
            idx += 1
        return full_note[:idx]

    def _extract_note_octave(self, full_note):
        """Extract octave number from note+octave string."""
        if not full_note:
            return None
        idx = 0
        while idx < len(full_note) and not (full_note[idx].isdigit() or full_note[idx] == "-"):
            idx += 1
        if idx >= len(full_note):
            return None
        try:
            return int(full_note[idx:])
        except ValueError:
            return None
