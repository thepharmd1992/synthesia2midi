"""White-center edge correction learned from black-key residuals."""

import numpy as np


class BlackResidualWarpMixin:
    def _estimate_black_residual_samples(self, black_note_centers, white_center_by_note):
        """Collect black-key residual samples: observed black center minus predicted center."""
        if not black_note_centers or not white_center_by_note:
            return []

        black_to_white_neighbors = {
            "C#": ("C", "D"),
            "D#": ("D", "E"),
            "F#": ("F", "G"),
            "G#": ("G", "A"),
            "A#": ("A", "B"),
        }

        raw_samples = []
        for black_note, observed_center in black_note_centers.items():
            black_name = self._extract_note_name(black_note)
            octave = self._extract_note_octave(black_note)
            if octave is None or black_name not in black_to_white_neighbors:
                continue

            left_white_name, right_white_name = black_to_white_neighbors[black_name]
            left_white_note = f"{left_white_name}{octave}"
            right_white_note = f"{right_white_name}{octave}"
            if left_white_note not in white_center_by_note or right_white_note not in white_center_by_note:
                continue

            predicted_center = (white_center_by_note[left_white_note] + white_center_by_note[right_white_note]) / 2.0
            residual = float(observed_center) - float(predicted_center)
            raw_samples.append((float(predicted_center), residual, black_name))

        if len(raw_samples) < 3:
            return []

        class_values = {}
        for _, residual, black_name in raw_samples:
            class_values.setdefault(black_name, []).append(residual)
        class_bias = {
            black_name: float(np.median(values))
            for black_name, values in class_values.items()
            if values
        }

        # Remove note-class bias so the fit captures global edge trend rather than
        # intrinsic keyboard asymmetry (e.g., C#/D# vs F#/G# midpoint offsets).
        samples = [
            (x_pos, residual - class_bias.get(black_name, 0.0))
            for x_pos, residual, black_name in raw_samples
        ]
        return samples

    def _apply_black_residual_edge_warp(
        self,
        centers,
        width,
        step_estimate,
        black_note_centers,
        white_center_by_note,
    ):
        """
        Apply a smooth edge warp learned from black-key residuals.

        The correction is forced near zero in the center and grows toward edges.
        """
        if not centers or width <= 2 or step_estimate <= 0:
            return centers

        samples = self._estimate_black_residual_samples(black_note_centers, white_center_by_note)
        if len(samples) < 3:
            return centers

        x_samples = np.asarray([pt[0] for pt in samples], dtype=np.float64)
        residual_samples = np.asarray([pt[1] for pt in samples], dtype=np.float64)

        x_mid = (float(width) - 1.0) / 2.0
        half_width = max(1.0, (float(width) - 1.0) / 2.0)
        x_norm = (x_samples - x_mid) / half_width

        if len(samples) >= 5:
            degree = 2
        else:
            degree = 1

        try:
            coeffs = np.polyfit(x_norm, residual_samples, degree)
        except Exception:
            return centers

        poly = np.poly1d(coeffs)
        center_bias = float(poly(0.0))
        max_correction = min(6.0, max(2.0, step_estimate * 0.22))

        corrected = []
        for center in centers:
            x_n = (float(center) - x_mid) / half_width
            raw_delta = float(poly(x_n) - center_bias)
            edge_gain = min(1.0, abs(x_n) ** 1.15)
            delta = raw_delta * edge_gain
            delta = float(np.clip(delta, -max_correction, max_correction))
            corrected.append(float(center) + delta)

        # Enforce strict ordering and a minimum spacing to avoid post-warp inversions.
        min_gap = max(3.0, step_estimate * 0.35)
        corrected = sorted(corrected)
        for idx in range(1, len(corrected)):
            corrected[idx] = max(corrected[idx], corrected[idx - 1] + min_gap)

        right_limit = float(width - 1)
        overflow = corrected[-1] - right_limit
        if overflow > 0:
            corrected = [val - overflow for val in corrected]

        if corrected[0] < 0:
            shift = -corrected[0]
            corrected = [val + shift for val in corrected]

        for idx in range(1, len(corrected)):
            corrected[idx] = max(corrected[idx], corrected[idx - 1] + min_gap)

        if corrected[-1] > right_limit:
            compression = corrected[-1] - right_limit
            corrected[-1] = right_limit
            if len(corrected) > 1:
                spread = len(corrected) - 1
                for idx in range(spread):
                    backshift = compression * ((spread - idx) / spread)
                    corrected[idx] = max(0.0, corrected[idx] - backshift)

        final_centers = [float(np.clip(val, 0.0, right_limit)) for val in corrected]
        self.logger.debug(
            "Applied black residual edge warp: samples=%d degree=%d max_corr=%.2f",
            len(samples),
            degree,
            max_correction,
        )
        return final_centers
