"""High-level musical note assignment orchestration."""


class NoteAssignmentMixin:
    def assign_notes(self):
        """Assign musical notes using unified chromatic scanning from F# anchor"""
        if not self.black_keys or not self.white_keys:
            raise ValueError("Must detect keys first")

        self.logger.debug(f"\n=== Assigning Notes to {len(self.black_keys)} black + {len(self.white_keys)} white keys ===")

        if self.params.get("type_aware_assignment", False):
            self.key_notes = self._assign_notes_type_aware()
        else:
            # Find F# anchor using confident LSSL pattern detection
            f_sharp_position = self._find_confident_f_sharp_anchor()

            if f_sharp_position is None:
                self.logger.debug("Could not find confident F# anchor - using fallback assignment")
                self.key_notes = self._fallback_note_assignment()
            else:
                # Unified chromatic assignment using pixel-by-pixel scanning
                self.key_notes = self._assign_notes_chromatically_from_anchor(f_sharp_position)

        self._apply_white_post_assignment_adjustments()

        self.logger.debug(f"DEBUG: Total assigned keys: {len(self.key_notes)}")

        if self.key_notes:
            self.logger.debug("First 10 chromatic note assignments:")
            sorted_notes = sorted(self.key_notes.items())
            for i, (center_x, note_info) in enumerate(sorted_notes[:10]):
                self.logger.debug(f"  Key {i}: center_x={center_x}, note={note_info['note']}, type={note_info['type']}")

        self.logger.debug(f"Assigned notes to {len(self.key_notes)} keys")
        return self.key_notes

    def _assign_notes_type_aware(self):
        """Assign notes by key type, using black-key anchors and white-key scanning."""
        candidates = self._find_f_sharp_anchor_candidates()
        if not candidates:
            return self._fallback_note_assignment()

        fallback_notes = None

        for f_sharp_idx in candidates:
            black_notes = self._assign_black_key_notes(f_sharp_idx)
            white_notes, used_fallback = self._assign_white_key_notes_by_scanning(
                black_notes,
                return_fallback=True,
            )

            if not white_notes:
                continue

            combined = {**black_notes, **white_notes}
            if not used_fallback:
                return combined

            if fallback_notes is None:
                fallback_notes = combined

        if fallback_notes:
            return fallback_notes

        return self._fallback_note_assignment()

    def _assign_notes_chromatically_from_anchor(self, f_sharp_center_x):
        """Assign notes chromatically using pixel-by-pixel scanning from F# anchor"""
        self.logger.debug(f"Starting chromatic assignment from F# anchor at x={f_sharp_center_x}")

        # Create unified list of all key overlays (black + white) sorted by position
        all_overlays = []

        # Add black keys
        for black_key in self.black_keys:
            center_x = black_key[0] + black_key[2] // 2
            all_overlays.append({
                'center_x': center_x,
                'type': 'black',
                'box': black_key,
                'assigned': False
            })

        # Add white keys
        for white_key in self.white_keys:
            center_x = white_key[0] + white_key[2] // 2
            all_overlays.append({
                'center_x': center_x,
                'type': 'white',
                'box': white_key,
                'assigned': False
            })

        # Sort all overlays by center_x position
        all_overlays.sort(key=lambda k: k['center_x'])

        self.logger.debug(f"Total overlays to assign: {len(all_overlays)} (scanning from F# anchor)")

        # Find F# overlay in sorted list
        f_sharp_idx = None
        for i, overlay in enumerate(all_overlays):
            if abs(overlay['center_x'] - f_sharp_center_x) < 5:  # Close match to F# anchor
                f_sharp_idx = i
                break

        if f_sharp_idx is None:
            self.logger.debug("❌ Could not find F# overlay in sorted list")
            return {}

        self.logger.debug(f"F# anchor found at overlay index {f_sharp_idx}")

        # Chromatic note sequence (semitones)
        chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        f_sharp_note_idx = 6  # F# is at index 6 in chromatic sequence

        assigned_notes = {}

        # Assign F# anchor first
        octave = 0  # Start at octave 0, will adjust based on position
        all_overlays[f_sharp_idx]['assigned'] = True
        note_name = 'F#'

        # Determine if F# overlay is black or white to set correct type
        overlay_type = all_overlays[f_sharp_idx]['type']

        assigned_notes[f_sharp_center_x] = {
            'note': f'{note_name}{octave}',
            'type': overlay_type,
            'box': all_overlays[f_sharp_idx]['box']
        }

        self.logger.debug(f"✅ Assigned F# anchor: x={f_sharp_center_x}, note=F#{octave}, type={overlay_type}")

        # Scan rightward from F# anchor
        current_note_idx = f_sharp_note_idx
        current_octave = octave

        for i in range(f_sharp_idx + 1, len(all_overlays)):
            overlay = all_overlays[i]
            if overlay['assigned']:
                continue

            # Move to next chromatic note
            current_note_idx = (current_note_idx + 1) % 12
            if current_note_idx == 0:  # Wrapped around to C
                current_octave += 1

            note_name = chromatic_notes[current_note_idx]
            overlay['assigned'] = True

            assigned_notes[overlay['center_x']] = {
                'note': f'{note_name}{current_octave}',
                'type': overlay['type'],
                'box': overlay['box']
            }

            self.logger.debug(f"→ Right scan: x={overlay['center_x']}, note={note_name}{current_octave}, type={overlay['type']}")

        # Scan leftward from F# anchor
        current_note_idx = f_sharp_note_idx
        current_octave = octave

        for i in range(f_sharp_idx - 1, -1, -1):
            overlay = all_overlays[i]
            if overlay['assigned']:
                continue

            # Move to previous chromatic note
            current_note_idx = (current_note_idx - 1) % 12
            if current_note_idx == 11:  # Wrapped around to B from C
                current_octave -= 1

            note_name = chromatic_notes[current_note_idx]
            overlay['assigned'] = True

            assigned_notes[overlay['center_x']] = {
                'note': f'{note_name}{current_octave}',
                'type': overlay['type'],
                'box': overlay['box']
            }

            self.logger.debug(f"← Left scan: x={overlay['center_x']}, note={note_name}{current_octave}, type={overlay['type']}")

        self.logger.debug(f"✅ Chromatic assignment complete: {len(assigned_notes)} keys assigned")
        return assigned_notes

    def _fallback_note_assignment(self):
        """Fallback note assignment when F# anchor fails"""
        notes = {}

        # Simple chromatic assignment starting from C4
        chromatic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Combine all keys and sort by position
        all_keys = []
        for bk in self.black_keys:
            center_x = bk[0] + bk[2] // 2
            all_keys.append((center_x, 'black', bk))

        for wk in self.white_keys:
            center_x = wk[0] + wk[2] // 2
            all_keys.append((center_x, 'white', wk))

        all_keys.sort()

        # Assign notes starting from C4
        start_octave = 4
        for i, (center_x, key_type, box) in enumerate(all_keys):
            note_idx = i % 12
            octave = start_octave + (i // 12)

            note_name = chromatic_notes[note_idx]
            full_note = f"{note_name}{octave}"

            notes[center_x] = {
                'note': full_note,
                'type': key_type,
                'box': box
            }

        return notes
