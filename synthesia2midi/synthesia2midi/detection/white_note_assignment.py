"""White-key note assignment and post-assignment overlay adjustment."""

import numpy as np


class WhiteNoteAssignmentMixin:
    def _apply_white_post_assignment_adjustments(self):
        """Apply post-assignment geometry adjustments for white-key overlays."""
        if not self.key_notes or not self.keyboard_region:
            return

        # Use current white key ordering to clamp shifts between neighboring keys.
        sorted_whites = sorted(
            (
                {
                    "center_x": center_x,
                    "note_info": note_info,
                    "x": int(note_info["box"][0]),
                    "y": int(note_info["box"][1]),
                    "w": int(note_info["box"][2]),
                    "h": int(note_info["box"][3]),
                    "note_name": self._extract_note_name(note_info.get("note", "")),
                }
                for center_x, note_info in self.key_notes.items()
                if note_info.get("type") == "white" and note_info.get("box")
            ),
            key=lambda item: item["x"],
        )

        if not sorted_whites:
            return

        _, _, roi_left_x, roi_right_x = self.keyboard_region
        roi_width = max(1, int(roi_right_x - roi_left_x))

        # Lock D/G/A onto midpoint between adjacent black keys.
        midpoint_pairs = {
            "D": ("C#", "D#"),
            "G": ("F#", "G#"),
            "A": ("G#", "A#"),
        }
        black_center_by_note = {
            str(note_info.get("note", "")): int(center_x)
            for center_x, note_info in self.key_notes.items()
            if note_info.get("type") == "black"
        }

        midpoint_count = 0
        for idx, item in enumerate(sorted_whites):
            note_name = item["note_name"]
            if note_name not in midpoint_pairs:
                continue

            octave = self._extract_note_octave(item["note_info"].get("note", ""))
            if octave is None:
                continue

            left_note, right_note = midpoint_pairs[note_name]
            left_note = f"{left_note}{octave}"
            right_note = f"{right_note}{octave}"
            if left_note not in black_center_by_note or right_note not in black_center_by_note:
                continue

            target_center = int(round((black_center_by_note[left_note] + black_center_by_note[right_note]) / 2.0))
            desired_x = target_center - (item["w"] // 2)

            min_x = 0
            max_x = max(0, roi_width - item["w"])
            if idx > 0:
                prev = sorted_whites[idx - 1]
                min_x = max(min_x, prev["x"] + prev["w"] + 1)
            if idx < (len(sorted_whites) - 1):
                nxt = sorted_whites[idx + 1]
                max_x = min(max_x, nxt["x"] - item["w"] - 1)

            if max_x < min_x:
                continue

            new_x = int(max(min_x, min(max_x, desired_x)))
            if new_x == item["x"]:
                continue

            note_info = item["note_info"]
            note_info["box"] = (new_x, item["y"], item["w"], item["h"])
            item["x"] = new_x
            midpoint_count += 1

        # Re-sort after midpoint locking before manual edge-tail shift.
        sorted_whites.sort(key=lambda item: item["x"])

        left_ticks = int(round(float(self.params.get("white_edge_left_shift_ticks", 0))))
        right_ticks = int(round(float(self.params.get("white_edge_right_shift_ticks", 0))))
        edge_shift_count = 0
        if (left_ticks != 0 or right_ticks != 0) and len(sorted_whites) >= 3:
            edge_count = max(1, int(round(len(sorted_whites) * 0.20)))
            if (2 * edge_count) >= len(sorted_whites):
                edge_count = max(1, (len(sorted_whites) - 1) // 2)

            median_width = float(np.median([item["w"] for item in sorted_whites]))
            px_per_tick = median_width * 0.05
            left_edge_delta_px = -float(left_ticks) * px_per_tick
            right_edge_delta_px = float(right_ticks) * px_per_tick
            inner_weight_floor = 0.01
            edge_falloff_power = 6.0

            def _edge_weight(edge_progress: float) -> float:
                """Map edge progress to shift weight with a 1% floor at inner tail edge."""
                base_min = 1.0 / float(max(1, edge_count))
                if edge_progress <= base_min:
                    return inner_weight_floor
                normalized = (edge_progress - base_min) / max(1e-6, 1.0 - base_min)
                normalized = float(np.clip(normalized, 0.0, 1.0))
                curved = normalized ** edge_falloff_power
                return inner_weight_floor + ((1.0 - inner_weight_floor) * curved)

            desired_positions = []
            for idx, item in enumerate(sorted_whites):
                shift_px = 0.0

                if idx < edge_count:
                    edge_progress = (edge_count - idx) / float(max(1, edge_count))
                    shift_px += left_edge_delta_px * _edge_weight(edge_progress)

                if idx >= len(sorted_whites) - edge_count:
                    right_idx = (len(sorted_whites) - 1) - idx
                    edge_progress = (edge_count - right_idx) / float(max(1, edge_count))
                    shift_px += right_edge_delta_px * _edge_weight(edge_progress)

                desired_x = int(round(item["x"] + shift_px))
                max_x = max(0, roi_width - item["w"])
                desired_x = max(0, min(max_x, desired_x))
                desired_positions.append(desired_x)

            adjusted_positions = list(desired_positions)
            for idx in range(1, len(sorted_whites)):
                prev_item = sorted_whites[idx - 1]
                min_x = adjusted_positions[idx - 1] + prev_item["w"] + 1
                if adjusted_positions[idx] < min_x:
                    adjusted_positions[idx] = min_x

            for idx in range(len(sorted_whites) - 2, -1, -1):
                cur_item = sorted_whites[idx]
                max_x = adjusted_positions[idx + 1] - cur_item["w"] - 1
                if adjusted_positions[idx] > max_x:
                    adjusted_positions[idx] = max_x

            for idx, item in enumerate(sorted_whites):
                max_x = max(0, roi_width - item["w"])
                adjusted_positions[idx] = max(0, min(max_x, adjusted_positions[idx]))

            for idx in range(1, len(sorted_whites)):
                prev_item = sorted_whites[idx - 1]
                min_x = adjusted_positions[idx - 1] + prev_item["w"] + 1
                if adjusted_positions[idx] < min_x:
                    adjusted_positions[idx] = min_x

            for idx in range(len(sorted_whites) - 2, -1, -1):
                cur_item = sorted_whites[idx]
                max_x = adjusted_positions[idx + 1] - cur_item["w"] - 1
                if adjusted_positions[idx] > max_x:
                    adjusted_positions[idx] = max_x

            for idx, item in enumerate(sorted_whites):
                new_x = adjusted_positions[idx]
                if new_x == item["x"]:
                    continue
                item["x"] = new_x
                item["note_info"]["box"] = (new_x, item["y"], item["w"], item["h"])
                edge_shift_count += 1

        if midpoint_count > 0 or edge_shift_count > 0:
            self.logger.debug(
                "Applied white post-adjustments: midpoint=%d edge_shift=%d left_ticks=%d right_ticks=%d",
                midpoint_count,
                edge_shift_count,
                left_ticks,
                right_ticks,
            )

    def _assign_white_key_notes_by_scanning(self, black_notes, return_fallback=False):
        """Assign white key notes by scanning from F# anchor"""
        white_notes = {}

        # Find F# position
        f_sharp_center = None
        f_sharp_note = None

        for center, note_info in black_notes.items():
            if note_info['note'].startswith('F#'):
                f_sharp_center = center
                f_sharp_note = note_info['note']
                break

        if f_sharp_center is None:
            white_notes = self._fallback_white_assignment()
            if return_fallback:
                return white_notes, True
            return white_notes

        # White key pattern starting from F (before F#)
        white_pattern = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

        # Extract octave from F# note
        f_sharp_octave = int(f_sharp_note[2:])

        # F comes before F# in the same octave
        f_note = f'F{f_sharp_octave}'
        f_pattern_idx = 3  # F is at index 3 in white pattern

        # Sort white keys by position
        sorted_white_keys = sorted(self.white_keys, key=lambda k: k[0])

        # Find the white key closest to and left of F#
        f_key_idx = None
        min_distance = float('inf')

        for i, white_key in enumerate(sorted_white_keys):
            white_center = white_key[0] + white_key[2] // 2
            if white_center < f_sharp_center:
                distance = f_sharp_center - white_center
                if distance < min_distance:
                    min_distance = distance
                    f_key_idx = i

        if f_key_idx is None:
            white_notes = self._fallback_white_assignment()
            if return_fallback:
                return white_notes, True
            return white_notes

        # Assign notes starting from F
        for i, white_key in enumerate(sorted_white_keys):
            # Calculate position relative to F
            relative_pos = i - f_key_idx

            # Calculate pattern index and octave
            pattern_idx = (f_pattern_idx + relative_pos) % 7
            octave_offset = (f_pattern_idx + relative_pos) // 7
            octave = f_sharp_octave + octave_offset

            # Get note name
            note_name = white_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"

            # Store with center position as key
            center_x = white_key[0] + white_key[2] // 2
            white_notes[center_x] = {
                'note': full_note,
                'type': 'white',
                'box': white_key
            }

        if return_fallback:
            return white_notes, False
        return white_notes

    def _fallback_white_assignment(self):
        """Fallback white key assignment"""
        white_notes = {}
        white_pattern = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

        sorted_white_keys = sorted(self.white_keys, key=lambda k: k[0])

        for i, white_key in enumerate(sorted_white_keys):
            pattern_idx = i % 7
            octave = 4 + (i // 7)

            note_name = white_pattern[pattern_idx]
            full_note = f"{note_name}{octave}"

            center_x = white_key[0] + white_key[2] // 2
            white_notes[center_x] = {
                'note': full_note,
                'type': 'white',
                'box': white_key
            }

        return white_notes
