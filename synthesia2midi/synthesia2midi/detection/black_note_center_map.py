"""Build note-center maps from detected black-key geometry."""


class BlackNoteCenterMapMixin:
    def _score_black_note_center_map(self, note_centers):
        """Score black-note maps by how many complete D-anchored octave segments they provide."""
        if not note_centers:
            return 0.0

        octaves = set()
        for note in note_centers.keys():
            octave = self._extract_note_octave(note)
            if octave is not None:
                octaves.add(octave)

        pair_count = 0
        full_segment_count = 0
        for octave in sorted(octaves):
            has_pair = f"C#{octave}" in note_centers and f"D#{octave}" in note_centers
            if has_pair:
                pair_count += 1
            has_full = (
                has_pair
                and f"F#{octave}" in note_centers
                and f"G#{octave}" in note_centers
                and f"A#{octave}" in note_centers
                and f"C#{octave + 1}" in note_centers
                and f"D#{octave + 1}" in note_centers
            )
            if has_full:
                full_segment_count += 1

        return (full_segment_count * 10.0) + pair_count + (len(note_centers) * 0.01)

    def _build_black_note_center_map(self):
        """Assign black-note labels for geometry solving and return note->center map."""
        if not self.black_keys:
            return {}

        self.black_keys = sorted(self.black_keys, key=lambda key: key[0])
        candidates = self._find_f_sharp_anchor_candidates()
        if not candidates:
            return {}

        best_map = {}
        best_score = -1.0
        for f_sharp_idx in candidates:
            try:
                black_notes = self._assign_black_key_notes(f_sharp_idx)
            except Exception:
                continue

            note_centers = {}
            for center_x, note_info in black_notes.items():
                note = str(note_info.get("note", "")).strip()
                if not note:
                    continue
                note_centers[note] = float(center_x)

            score = self._score_black_note_center_map(note_centers)
            if score > best_score:
                best_score = score
                best_map = note_centers

        return best_map
