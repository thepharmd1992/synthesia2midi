"""Shared geometry helpers for the manual piano keyboard detector."""

import numpy as np


class DetectorGeometryMixin:
    def _add_overlay_padding(self, start_x, y, width, height, padding_percent=None):
        """Add padding to overlay by shrinking inward from left and right sides"""
        if padding_percent is None:
            padding_percent = self.params.get("padding_percent", 0.25)
        padding_pixels = int(width * padding_percent)
        new_start_x = start_x + padding_pixels
        new_width = width - (2 * padding_pixels)
        return new_start_x, y, new_width, height

    def _runs_from_mask(self, mask: np.ndarray, *, min_width: int = 2):
        runs = []
        in_run = False
        start = 0
        for x, v in enumerate(mask):
            if v and not in_run:
                in_run = True
                start = x
            elif (not v) and in_run:
                end = x - 1
                if (end - start + 1) >= min_width:
                    runs.append((start, end))
                in_run = False
        if in_run:
            end = len(mask) - 1
            if (end - start + 1) >= min_width:
                runs.append((start, end))
        return runs

    def _split_span_evenly(self, x_left, x_right, count, min_key_width):
        """Split an overly wide span into evenly sized key candidates."""
        if count <= 1:
            return [(x_left, x_right)]

        span_width = x_right - x_left + 1
        if span_width <= 0:
            return []

        count = max(1, int(count))
        spans = []
        for idx in range(count):
            seg_left = x_left + int(round((idx * span_width) / count))
            seg_right = x_left + int(round(((idx + 1) * span_width) / count)) - 1
            if seg_right < seg_left:
                continue
            if (seg_right - seg_left + 1) >= min_key_width:
                spans.append((seg_left, seg_right))
        return spans
