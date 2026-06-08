"""Manual keyboard fit session model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

from synthesia2midi.app_config import NOTE_NAMES_SHARP


WHITE_NOTE_NAMES = {name for name in NOTE_NAMES_SHARP if "♯" not in name and "♭" not in name}
HORIZONTAL_SAFE_INSET_FRACTION = 0.10
VERTICAL_SAFE_INSET_FRACTION = 0.15


@dataclass
class ManualFitParams:
    group_dx: float = 0.0
    group_dy: float = 0.0
    keyboard_width_delta: float = 0.0
    keyboard_top_delta: float = 0.0
    left_edge_drift: float = 0.0
    right_edge_drift: float = 0.0
    black_width_delta: float = 0.0


CONTROL_PARAM_NAMES = (
    "keyboard_width_delta",
    "keyboard_top_delta",
    "left_edge_drift",
    "right_edge_drift",
    "black_width_delta",
)


@dataclass(frozen=True)
class OverlayGeometry:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class DetectionRegion:
    top: float
    bottom: float


@dataclass
class OverlayOverride:
    x_delta: float = 0.0
    y_delta: float = 0.0
    width_delta: float = 0.0
    height_delta: float = 0.0


class ManualKeyboardFitSession:
    def __init__(self, app_state):
        self.app_state = app_state
        self.params = ManualFitParams()
        self._previous_unsaved_changes = bool(app_state.unsaved_changes)
        self._previous_octave_transpose = int(getattr(app_state.midi, "octave_transpose", 0))
        self._baseline: Dict[int, OverlayGeometry] = {
            overlay.key_id: OverlayGeometry(
                float(overlay.x),
                float(overlay.y),
                float(overlay.width),
                float(overlay.height),
            )
            for overlay in app_state.overlays
        }
        self._overrides: Dict[int, OverlayOverride] = {}
        self._bounds = self._calculate_bounds(self._baseline.values())
        self._default_regions = self._calculate_default_regions()
        self._custom_regions: Dict[str, DetectionRegion] = {}

    def update_params(self, params: ManualFitParams) -> None:
        self.params = params
        self.apply_preview()

    def update_control_params(self, params: ManualFitParams) -> None:
        for name in CONTROL_PARAM_NAMES:
            setattr(self.params, name, getattr(params, name))
        self.apply_preview()

    def set_param(self, name: str, value: float) -> None:
        if not hasattr(self.params, name):
            raise AttributeError(f"Unknown manual fit parameter: {name}")
        setattr(self.params, name, float(value))
        self.apply_preview()

    def translate_group(self, dx: float, dy: float) -> None:
        self.params.group_dx += float(dx)
        self.params.group_dy += float(dy)
        self.apply_preview()

    def reset_position(self) -> None:
        self.params.group_dx = 0.0
        self.params.group_dy = 0.0
        self.apply_preview()

    def set_detection_region(self, key_type: str, top: float, bottom: float) -> None:
        if key_type not in {"white", "black"}:
            raise ValueError(f"Unknown detection region type: {key_type}")
        region_top = min(float(top), float(bottom)) - self.params.group_dy
        region_bottom = max(float(top), float(bottom)) - self.params.group_dy
        self._custom_regions[key_type] = DetectionRegion(region_top, region_bottom)
        self.apply_preview()

    def detection_region_guides(self) -> Dict[str, DetectionRegion]:
        return {
            "white": self._region_for_type("white"),
            "black": self._region_for_type("black"),
        }

    def set_octave_transpose(self, value: int) -> None:
        self.app_state.midi.octave_transpose = int(value)
        self.app_state.unsaved_changes = True

    def current_octave_transpose(self) -> int:
        return int(getattr(self.app_state.midi, "octave_transpose", 0))

    def move_single_overlay_by_index(self, overlay_index: int, new_x: float, new_y: float) -> bool:
        if not 0 <= overlay_index < len(self.app_state.overlays):
            return False

        overlay = self.app_state.overlays[overlay_index]
        base = self._transformed_rect_without_override(overlay.key_id)
        if base is None:
            return False

        self._overrides[overlay.key_id] = OverlayOverride(
            x_delta=float(new_x) - base.x,
            y_delta=float(new_y) - base.y,
            width_delta=float(overlay.width) - base.width,
            height_delta=float(overlay.height) - base.height,
        )
        self.apply_preview()
        return True

    def resize_single_overlay_by_index(
        self,
        overlay_index: int,
        new_x: float,
        new_y: float,
        new_width: float,
        new_height: float,
    ) -> bool:
        if not 0 <= overlay_index < len(self.app_state.overlays):
            return False

        overlay = self.app_state.overlays[overlay_index]
        base = self._transformed_rect_without_override(overlay.key_id)
        if base is None:
            return False

        self._overrides[overlay.key_id] = OverlayOverride(
            x_delta=float(new_x) - base.x,
            y_delta=float(new_y) - base.y,
            width_delta=float(new_width) - base.width,
            height_delta=float(new_height) - base.height,
        )
        self.apply_preview()
        return True

    def clear_override_for_key_id(self, key_id: int) -> bool:
        if key_id not in self._overrides:
            return False
        del self._overrides[key_id]
        self.apply_preview()
        return True

    def clear_selected_override(self) -> bool:
        selected_key_id = getattr(self.app_state.ui, "selected_overlay_id", None)
        if selected_key_id is None:
            return False
        return self.clear_override_for_key_id(int(selected_key_id))

    def reset_all(self) -> None:
        self.params = ManualFitParams()
        self._overrides.clear()
        self._custom_regions.clear()
        self.app_state.midi.octave_transpose = self._previous_octave_transpose
        self.apply_preview()

    def cancel(self) -> None:
        self._restore_baseline()
        self.app_state.midi.octave_transpose = self._previous_octave_transpose
        self.app_state.unsaved_changes = self._previous_unsaved_changes

    def apply(self) -> None:
        self.app_state.unsaved_changes = True

    def overridden_key_ids(self) -> Set[int]:
        return set(self._overrides)

    def apply_preview(self) -> None:
        for overlay in self.app_state.overlays:
            rect = self._transformed_rect_without_override(overlay.key_id)
            if rect is None:
                continue
            override = self._overrides.get(overlay.key_id)
            if override is not None:
                rect = OverlayGeometry(
                    rect.x + override.x_delta,
                    rect.y + override.y_delta,
                    max(1.0, rect.width + override.width_delta),
                    max(1.0, rect.height + override.height_delta),
                )
            overlay.x = rect.x
            overlay.y = rect.y
            overlay.width = rect.width
            overlay.height = rect.height
        self.app_state.unsaved_changes = True

    def _restore_baseline(self) -> None:
        for overlay in self.app_state.overlays:
            baseline = self._baseline.get(overlay.key_id)
            if baseline is None:
                continue
            overlay.x = baseline.x
            overlay.y = baseline.y
            overlay.width = baseline.width
            overlay.height = baseline.height

    def _transformed_rect_without_override(self, key_id: int) -> OverlayGeometry | None:
        baseline = self._baseline.get(key_id)
        if baseline is None:
            return None

        span = max(1.0, self._bounds[2])
        target_span = max(1.0, span + self.params.keyboard_width_delta)
        scale = target_span / span
        left, _right, _span, center = self._bounds

        baseline_center_x = baseline.x + baseline.width / 2
        norm = (baseline_center_x - left) / span
        scaled_width = max(1.0, baseline.width * scale)
        scaled_center_x = center + ((baseline_center_x - center) * scale)
        edge_shift = (self.params.left_edge_drift * (1.0 - norm)) + (
            self.params.right_edge_drift * norm
        )

        x = scaled_center_x - scaled_width / 2 + self.params.group_dx + edge_shift
        width = scaled_width
        top = baseline.y + self.params.group_dy
        bottom = baseline.y + baseline.height + self.params.group_dy

        overlay = next((candidate for candidate in self.app_state.overlays if candidate.key_id == key_id), None)
        is_white = overlay is not None and overlay.note_name_in_octave in WHITE_NOTE_NAMES
        if is_white:
            top, bottom = self._safe_region_bounds("white")
        else:
            top, bottom = self._safe_region_bounds("black")
            width = max(1.0, width + self.params.black_width_delta)
            x = (scaled_center_x + self.params.group_dx + edge_shift) - width / 2

        x, width = self._apply_fractional_x_inset(x, width)

        if bottom < top + 1.0:
            bottom = top + 1.0

        return OverlayGeometry(x, top, width, bottom - top)

    def _safe_region_bounds(self, key_type: str) -> Tuple[float, float]:
        region = self._region_for_type(key_type)
        top = region.top + self.params.group_dy + self.params.keyboard_top_delta
        bottom = region.bottom + self.params.group_dy
        region_height = max(1.0, bottom - top)
        vertical_inset = region_height * VERTICAL_SAFE_INSET_FRACTION
        return top + vertical_inset, bottom - vertical_inset

    def _region_for_type(self, key_type: str) -> DetectionRegion:
        return self._custom_regions.get(key_type) or self._default_regions[key_type]

    @staticmethod
    def _apply_fractional_x_inset(x: float, width: float) -> Tuple[float, float]:
        inset = width * HORIZONTAL_SAFE_INSET_FRACTION
        safe_inset = max(0.0, min(inset, (width - 1.0) / 2.0))
        return x + safe_inset, max(1.0, width - (2.0 * safe_inset))

    def _calculate_default_regions(self) -> Dict[str, DetectionRegion]:
        white_geometries = []
        black_geometries = []
        for overlay in self.app_state.overlays:
            geometry = self._baseline.get(overlay.key_id)
            if geometry is None:
                continue
            if overlay.note_name_in_octave in WHITE_NOTE_NAMES:
                white_geometries.append(geometry)
            else:
                black_geometries.append(geometry)
        return {
            "white": self._calculate_vertical_region(white_geometries or self._baseline.values()),
            "black": self._calculate_vertical_region(black_geometries or self._baseline.values()),
        }

    @staticmethod
    def _calculate_vertical_region(geometries: Iterable[OverlayGeometry]) -> DetectionRegion:
        geometries = list(geometries)
        if not geometries:
            return DetectionRegion(0.0, 1.0)
        top = min(geometry.y for geometry in geometries)
        bottom = max(geometry.y + geometry.height for geometry in geometries)
        return DetectionRegion(top, max(top + 1.0, bottom))

    @staticmethod
    def _calculate_bounds(geometries: Iterable[OverlayGeometry]) -> Tuple[float, float, float, float]:
        geometries = list(geometries)
        if not geometries:
            return 0.0, 1.0, 1.0, 0.5
        left = min(geometry.x for geometry in geometries)
        right = max(geometry.x + geometry.width for geometry in geometries)
        span = max(1.0, right - left)
        center = left + span / 2
        return left, right, span, center
