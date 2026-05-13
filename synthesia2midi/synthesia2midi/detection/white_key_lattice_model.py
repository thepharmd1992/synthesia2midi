"""D-anchored white-key center estimation model."""

import numpy as np


class WhiteKeyLatticeModelMixin:
    def _estimate_white_centers_from_d_lattice(self, black_note_centers, width):
        """
        Estimate white-key centers from D anchors.

        For each D(o)->D(o+1) span:
        - derive local 3-step D->G and A->D spacing from black geometry
        - place E/F/G and A/B/C as normalized local fractions of the span
        """
        c_sharp_by_octave = {}
        d_sharp_by_octave = {}
        f_sharp_by_octave = {}
        g_sharp_by_octave = {}
        a_sharp_by_octave = {}

        for note, center in black_note_centers.items():
            note_name = self._extract_note_name(note)
            octave = self._extract_note_octave(note)
            if octave is None:
                continue
            if note_name == "C#":
                c_sharp_by_octave[octave] = float(center)
            elif note_name == "D#":
                d_sharp_by_octave[octave] = float(center)
            elif note_name == "F#":
                f_sharp_by_octave[octave] = float(center)
            elif note_name == "G#":
                g_sharp_by_octave[octave] = float(center)
            elif note_name == "A#":
                a_sharp_by_octave[octave] = float(center)

        d_anchor_by_octave = {}
        for octave in sorted(set(c_sharp_by_octave.keys()) & set(d_sharp_by_octave.keys())):
            d_anchor_by_octave[octave] = (c_sharp_by_octave[octave] + d_sharp_by_octave[octave]) / 2.0

        segment_data = []
        for octave in sorted(d_anchor_by_octave.keys()):
            next_octave = octave + 1
            if next_octave not in d_anchor_by_octave:
                continue

            d0 = float(d_anchor_by_octave[octave])
            d1 = float(d_anchor_by_octave[next_octave])
            span = d1 - d0
            if span < 12.0:
                continue

            if octave in f_sharp_by_octave and octave in g_sharp_by_octave:
                g_anchor = (f_sharp_by_octave[octave] + g_sharp_by_octave[octave]) / 2.0
            else:
                g_anchor = d0 + (span * (3.0 / 7.0))

            if octave in g_sharp_by_octave and octave in a_sharp_by_octave:
                a_anchor = (g_sharp_by_octave[octave] + a_sharp_by_octave[octave]) / 2.0
            else:
                a_anchor = d0 + (span * (4.0 / 7.0))

            # Keep anchors in plausible octave-relative windows.
            g_anchor = float(np.clip(g_anchor, d0 + (span * 0.25), d0 + (span * 0.60)))
            a_anchor = float(np.clip(a_anchor, d0 + (span * 0.40), d0 + (span * 0.82)))
            if a_anchor <= g_anchor + (span * 0.08):
                a_anchor = min(d1 - (span * 0.10), g_anchor + (span * 0.16))

            left_step_raw = (g_anchor - d0) / 3.0
            right_step_raw = (d1 - a_anchor) / 3.0
            if left_step_raw <= 0:
                left_step_raw = span / 7.0
            if right_step_raw <= 0:
                right_step_raw = span / 7.0

            segment_data.append(
                {
                    "octave": octave,
                    "d0": d0,
                    "d1": d1,
                    "span": span,
                    "left_raw": left_step_raw,
                    "right_raw": right_step_raw,
                }
            )

        if not segment_data:
            return [], 0.0, {}

        left_raw_values = [seg["left_raw"] for seg in segment_data]
        right_raw_values = [seg["right_raw"] for seg in segment_data]
        left_median = float(np.median(left_raw_values)) if left_raw_values else 0.0
        right_median = float(np.median(right_raw_values)) if right_raw_values else 0.0

        white_center_by_note = {}

        def set_center(note_key, value):
            val = float(value)
            if note_key in white_center_by_note:
                white_center_by_note[note_key] = (white_center_by_note[note_key] + val) / 2.0
            else:
                white_center_by_note[note_key] = val

        for octave, d_center in d_anchor_by_octave.items():
            set_center(f"D{octave}", d_center)

        for seg in segment_data:
            octave = int(seg["octave"])
            span = float(seg["span"])
            d0 = float(seg["d0"])
            d1 = float(seg["d1"])

            left_step = float(seg["left_raw"])
            right_step = float(seg["right_raw"])
            if left_median > 0:
                left_step = float(np.clip(left_step, left_median * 0.78, left_median * 1.22))
            if right_median > 0:
                right_step = float(np.clip(right_step, right_median * 0.78, right_median * 1.22))

            # Physical guard rails for 7 white steps inside one D->D octave span.
            min_step = span * 0.10
            max_step = span * 0.20
            left_step = float(np.clip(left_step, min_step, max_step))
            right_step = float(np.clip(right_step, min_step, max_step))

            # If either side is too distorted, fall back to equal 1/7 spacing.
            if (left_step * 3.0) + (right_step * 3.0) > (span * 0.95):
                equal_step = span / 7.0
                left_step = equal_step
                right_step = equal_step

            e_center = d0 + left_step
            f_center = d0 + (left_step * 2.0)
            g_center = d0 + (left_step * 3.0)
            a_center = d1 - (right_step * 3.0)
            b_center = a_center + right_step
            c_next_center = a_center + (right_step * 2.0)

            if not (d0 < e_center < f_center < g_center < a_center < b_center < c_next_center < d1):
                equal_step = span / 7.0
                e_center = d0 + equal_step
                f_center = d0 + (equal_step * 2.0)
                g_center = d0 + (equal_step * 3.0)
                a_center = d0 + (equal_step * 4.0)
                b_center = d0 + (equal_step * 5.0)
                c_next_center = d0 + (equal_step * 6.0)

            set_center(f"E{octave}", e_center)
            set_center(f"F{octave}", f_center)
            set_center(f"G{octave}", g_center)
            set_center(f"A{octave}", a_center)
            set_center(f"B{octave}", b_center)
            set_center(f"C{octave + 1}", c_next_center)

        raw_centers = sorted(white_center_by_note.values())
        if len(raw_centers) < 4:
            return [], 0.0, {}

        center_diffs = [raw_centers[i + 1] - raw_centers[i] for i in range(len(raw_centers) - 1)]
        center_diffs = [diff for diff in center_diffs if diff > 1.0]
        step_estimate = float(np.median(center_diffs)) if center_diffs else 0.0
        if step_estimate <= 0:
            return [], 0.0, {}

        min_center_gap = max(3.0, step_estimate * 0.40)
        deduped_centers = []
        for center in raw_centers:
            if deduped_centers and abs(center - deduped_centers[-1]) < min_center_gap:
                deduped_centers[-1] = (deduped_centers[-1] + center) / 2.0
            else:
                deduped_centers.append(center)

        if len(deduped_centers) < 3:
            return [], 0.0, {}

        # Extend to ROI edges if D-anchored spans do not cover cropped ends.
        # Use ceil (not round) so edge gaps don't leave a large unsolved span,
        # which can trigger unstable guided splits and indexing drift.
        left_room = deduped_centers[0] - (step_estimate * 0.5)
        right_room = (float(width - 1) - deduped_centers[-1]) - (step_estimate * 0.5)
        extra_left = max(0, int(np.ceil(left_room / step_estimate)))
        extra_right = max(0, int(np.ceil(right_room / step_estimate)))
        max_extra = max(0, int(np.ceil(float(width) / max(step_estimate, 1.0))) + 2)
        extra_left = min(extra_left, max_extra)
        extra_right = min(extra_right, max_extra)

        for _ in range(extra_left):
            deduped_centers.insert(0, deduped_centers[0] - step_estimate)
        for _ in range(extra_right):
            deduped_centers.append(deduped_centers[-1] + step_estimate)

        final_centers = []
        for center in deduped_centers:
            clamped = float(np.clip(center, 0.0, float(width - 1)))
            if final_centers and abs(clamped - final_centers[-1]) < min_center_gap:
                continue
            final_centers.append(clamped)

        return final_centers, step_estimate, dict(white_center_by_note)
