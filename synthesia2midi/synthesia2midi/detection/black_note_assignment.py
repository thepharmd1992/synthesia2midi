"""Black-key anchor finding and note assignment."""


class BlackNoteAssignmentMixin:
    def _find_confident_f_sharp_anchor(self):
        """Find F# anchor by locating confident LSSL pattern (3-black-key group)"""
        if len(self.black_keys) < 3:
            self.logger.debug("Not enough black keys to find F# anchor")
            return None

        self.logger.debug("Scanning left-to-right for confident LSSL patterns...")

        # Calculate gaps between consecutive black keys
        gaps = []
        for i in range(len(self.black_keys) - 1):
            gap = self.black_keys[i+1][0] - (self.black_keys[i][0] + self.black_keys[i][2])
            gaps.append(gap)

        # Find median gap to distinguish small from large gaps
        median_gap = sorted(gaps)[len(gaps)//2]
        gap_threshold = median_gap * 1.4

        self.logger.debug(f"Gap analysis: median={median_gap:.1f}, threshold={gap_threshold:.1f}")

        # Look for LSSL patterns (Large gap, then 3 black keys with Small-Small-Large gaps)
        for i in range(len(gaps) - 3):
            # Check for LSSL pattern starting at position i
            if (gaps[i] > gap_threshold and          # L: Large gap before group
                gaps[i+1] <= gap_threshold and       # S: Small gap (F# to G#)
                gaps[i+2] <= gap_threshold and       # S: Small gap (G# to A#)
                gaps[i+3] > gap_threshold):          # L: Large gap after group

                # Found confident LSSL pattern
                f_sharp_key_idx = i + 1  # F# is first key after the large gap
                f_sharp_key = self.black_keys[f_sharp_key_idx]
                f_sharp_center_x = f_sharp_key[0] + f_sharp_key[2] // 2

                self.logger.debug(f"✅ Found confident LSSL pattern at black key index {f_sharp_key_idx}")
                self.logger.debug(f"   F# anchor: center_x={f_sharp_center_x}, box={f_sharp_key}")
                self.logger.debug(f"   Gap sequence: {gaps[i]:.1f}(L) {gaps[i+1]:.1f}(S) {gaps[i+2]:.1f}(S) {gaps[i+3]:.1f}(L)")

                return f_sharp_center_x

        # Fallback: look for any SSL pattern (3 consecutive black keys)
        self.logger.debug("No confident LSSL found, looking for any SSL pattern...")
        for i in range(len(gaps) - 2):
            if (gaps[i] <= gap_threshold and         # S: Small gap
                gaps[i+1] <= gap_threshold and       # S: Small gap
                gaps[i+2] > gap_threshold):          # L: Large gap after

                f_sharp_key_idx = i
                f_sharp_key = self.black_keys[f_sharp_key_idx]
                f_sharp_center_x = f_sharp_key[0] + f_sharp_key[2] // 2

                self.logger.debug(f"⚠️ Fallback SSL pattern at index {f_sharp_key_idx}")
                self.logger.debug(f"   F# anchor (fallback): center_x={f_sharp_center_x}")

                return f_sharp_center_x

        self.logger.debug("❌ Could not find any F# anchor pattern")
        return None

    def _find_f_sharp_anchor_candidates(self):
        """Return candidate F# black-key indices, ordered by confidence."""
        if len(self.black_keys) < 3:
            return []

        gaps = []
        for i in range(len(self.black_keys) - 1):
            gap = self.black_keys[i + 1][0] - (self.black_keys[i][0] + self.black_keys[i][2])
            gaps.append(gap)

        median_gap = sorted(gaps)[len(gaps) // 2]
        gap_threshold = median_gap * 1.4

        candidates = []

        for i in range(len(gaps) - 3):
            if (
                gaps[i] > gap_threshold and
                gaps[i + 1] <= gap_threshold and
                gaps[i + 2] <= gap_threshold and
                gaps[i + 3] > gap_threshold
            ):
                candidates.append(i + 1)

        for i in range(len(gaps) - 2):
            if (
                gaps[i] <= gap_threshold and
                gaps[i + 1] <= gap_threshold and
                gaps[i + 2] > gap_threshold
            ):
                if i not in candidates:
                    candidates.append(i)

        return candidates

    def _assign_black_key_notes(self, f_sharp_idx):
        """Assign notes to black keys starting from F# anchor"""
        black_notes = {}

        # Black key pattern in chromatic sequence
        black_key_pattern = ['C#', 'D#', 'F#', 'G#', 'A#']

        # Calculate starting octave (A0 starts the 88-key piano)
        # F# is the 3rd black key in the pattern (index 2)
        pattern_position = 2  # F# position in pattern

        # Estimate octave based on anchor position in black-key groups
        base_octave = max(0, (f_sharp_idx - pattern_position) // 5)

        for i, black_key in enumerate(self.black_keys):
            # Calculate pattern index
            pattern_idx = (i - f_sharp_idx + pattern_position) % 5

            # Calculate octave
            octave = base_octave + ((i - f_sharp_idx + pattern_position) // 5)

            # Get note name
            note_name = black_key_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"

            # Store with center position as key
            center_x = black_key[0] + black_key[2] // 2
            black_notes[center_x] = {
                'note': full_note,
                'type': 'black',
                'box': black_key
            }

        return black_notes
