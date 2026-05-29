"""Modeless auto-detect tuning dialog controller."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt

from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog


class AutoDetectTuningController:
    """Owns auto-detect tuning dialog state, context caching, and preview apply flow."""

    def __init__(
        self,
        app,
        *,
        apply_template_styles_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.app = app
        self._auto_detect_tuning_dialog = None
        self._last_auto_detect_tuning_context: Optional[Dict[str, Any]] = None
        self._active_wizard = None
        self._on_dialog_finished_callback: Optional[Callable[[], None]] = None
        self._apply_template_styles_callback = apply_template_styles_callback

    @property
    def active_dialog(self):
        """Return the retained modeless dialog instance while it is open."""
        return self._auto_detect_tuning_dialog

    @property
    def cached_context(self) -> Optional[Dict[str, Any]]:
        if self._last_auto_detect_tuning_context is None:
            return None
        return self._clone_auto_detect_tuning_context(self._last_auto_detect_tuning_context)

    def set_apply_template_styles_callback(self, callback: Callable[[], None]) -> None:
        self._apply_template_styles_callback = callback

    def _clone_auto_detect_tuning_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return a defensive copy; callers mutate cached tuning packets independently."""
        cloned: Dict[str, Any] = dict(context)
        frame_rgb = context.get("frame_rgb")
        if frame_rgb is not None:
            cloned["frame_rgb"] = np.copy(frame_rgb)
        keyboard_roi = context.get("keyboard_roi")
        if keyboard_roi is not None:
            cloned["keyboard_roi"] = tuple(int(v) for v in keyboard_roi)
        cloned["fallback_used"] = bool(context.get("fallback_used", False))
        return cloned

    def cache_context(self, context: Dict[str, Any]) -> None:
        self._last_auto_detect_tuning_context = self._clone_auto_detect_tuning_context(context)

    def get_current_frame_rgb_for_tuning(self) -> Optional[np.ndarray]:
        frame_rgb = getattr(self.app.keyboard_canvas, "current_frame_rgb", None)
        if frame_rgb is not None:
            return np.copy(frame_rgb)

        frame_idx = self.app.app_state.video.current_frame_index
        if self.app.video_session is None or frame_idx is None:
            return None

        success, frame_bgr = self.app.video_session.get_frame(frame_idx)
        if not success or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def build_context_from_state(self) -> Optional[Dict[str, Any]]:
        if not self.app.app_state.overlays:
            return None

        frame_rgb = self.get_current_frame_rgb_for_tuning()
        if frame_rgb is None:
            return None

        valid_overlays = [
            overlay
            for overlay in self.app.app_state.overlays
            if overlay.width > 0 and overlay.height > 0
        ]
        if not valid_overlays:
            return None

        min_x = min(float(overlay.x) for overlay in valid_overlays)
        min_y = min(float(overlay.y) for overlay in valid_overlays)
        max_x = max(float(overlay.x) + float(overlay.width) for overlay in valid_overlays)
        max_y = max(float(overlay.y) + float(overlay.height) for overlay in valid_overlays)

        frame_h, frame_w = frame_rgb.shape[:2]
        x = max(0, int(np.floor(min_x)))
        y = max(0, int(np.floor(min_y)))
        right = min(frame_w, int(np.ceil(max_x)))
        bottom = min(frame_h, int(np.ceil(max_y)))
        width = right - x
        height = bottom - y
        if width <= 0 or height <= 0:
            return None

        detection_results: Dict[str, Any] = {
            "total_keys": int(self.app.app_state.midi.total_keys),
            "leftmost_note": self.app.app_state.midi.leftmost_note_name,
            "leftmost_octave": int(self.app.app_state.midi.leftmost_note_octave),
            "detected_keys": [],
        }
        if self._last_auto_detect_tuning_context is not None:
            cached_results = self._last_auto_detect_tuning_context.get("detection_results")
            if isinstance(cached_results, dict):
                detection_results = dict(cached_results)

        return {
            "frame_rgb": frame_rgb,
            "keyboard_roi": (x, y, width, height),
            "fallback_used": bool(
                self._last_auto_detect_tuning_context.get("fallback_used", False)
                if self._last_auto_detect_tuning_context is not None
                else False
            ),
            "detection_results": detection_results,
        }

    def resolve_context(self, wizard, *, use_wizard_context: bool) -> Optional[Dict[str, Any]]:
        if use_wizard_context and wizard is not None:
            wizard_context = wizard.get_auto_detect_tuning_context()
            if wizard_context:
                self.cache_context(wizard_context)
                return self._clone_auto_detect_tuning_context(wizard_context)

        if self._last_auto_detect_tuning_context is not None:
            return self._clone_auto_detect_tuning_context(self._last_auto_detect_tuning_context)

        state_context = self.build_context_from_state()
        if state_context is not None:
            self.cache_context(state_context)
            return self._clone_auto_detect_tuning_context(state_context)

        if not use_wizard_context and wizard is not None:
            wizard_context = wizard.get_auto_detect_tuning_context()
            if wizard_context:
                self.cache_context(wizard_context)
                return self._clone_auto_detect_tuning_context(wizard_context)

        return None

    def has_editable_context(self) -> bool:
        if self._last_auto_detect_tuning_context is not None:
            return True
        if not self.app.app_state.overlays:
            return False
        return self.get_current_frame_rgb_for_tuning() is not None

    def apply_preview_result(self, detection_results: Dict[str, Any]) -> bool:
        if self._active_wizard is None:
            return False

        applied = self._active_wizard.apply_auto_detect_results(detection_results)
        if not applied:
            return False

        wizard_context = self._active_wizard.get_auto_detect_tuning_context()
        if wizard_context:
            wizard_context = dict(wizard_context)
            wizard_context["detection_results"] = dict(detection_results)
            self.cache_context(wizard_context)
        elif self._last_auto_detect_tuning_context is not None:
            self._last_auto_detect_tuning_context["detection_results"] = dict(detection_results)

        if self._apply_template_styles_callback is not None:
            self._apply_template_styles_callback()
        self.app.app_state.ui.show_overlays = True
        self.app.show_overlays_action.setChecked(True)
        self.app.control_panel.convert_button.setEnabled(self.app.control_panel._can_convert())

        current_frame = self.app.app_state.video.current_frame_index
        if current_frame is not None:
            self.app.keyboard_canvas.display_frame(current_frame)
        else:
            self.app.keyboard_canvas.update()

        self.app.control_panel.update_controls_from_state()
        self.app.control_panel.update_selected_overlay_display()
        return True

    def open(
        self,
        wizard,
        *,
        use_wizard_context: bool = True,
        on_dialog_finished: Optional[Callable[[], None]] = None,
        dialog_factory: Optional[Callable[..., Any]] = None,
    ) -> bool:
        if wizard is None:
            return False

        context = self.resolve_context(wizard, use_wizard_context=use_wizard_context)
        if not context:
            logging.warning("Missing auto-detect tuning context; skipping tuning dialog")
            return False

        if self._auto_detect_tuning_dialog is not None:
            try:
                self._auto_detect_tuning_dialog.finished.disconnect(
                    self._on_auto_detect_tuning_dialog_finished
                )
            except Exception:
                pass
            self._auto_detect_tuning_dialog.close()
            self._auto_detect_tuning_dialog = None

        self._active_wizard = wizard
        self._on_dialog_finished_callback = on_dialog_finished
        factory = dialog_factory or AutoDetectTuningDialog
        dialog = factory(
            self.app,
            self.app.app_state,
            context["frame_rgb"],
            context["keyboard_roi"],
            initial_detection_results=context.get("detection_results"),
            fallback_used=bool(context.get("fallback_used", False)),
            apply_detection_callback=self.apply_preview_result,
        )
        dialog.setModal(False)
        dialog.setWindowModality(Qt.NonModal)
        dialog.finished.connect(self._on_auto_detect_tuning_dialog_finished)
        self._auto_detect_tuning_dialog = dialog

        # Position dialog toward the right side so it does not cover the keyboard area.
        screen = self.app.screen() if hasattr(self.app, "screen") else None
        if screen is not None:
            available = screen.availableGeometry()
            frame = dialog.frameGeometry()
            x = max(
                available.left() + 10,
                available.right() - frame.width() - 20,
            )
            y = min(
                max(available.top() + 40, self.app.geometry().top() + 20),
                max(available.top() + 10, available.bottom() - frame.height() - 20),
            )
            dialog.move(x, y)

        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return True

    def _on_auto_detect_tuning_dialog_finished(self, _result: int) -> None:
        self._auto_detect_tuning_dialog = None

        # Persist tuned params/overlays with the existing per-video save flow.
        if self.app.app_state.unsaved_changes and self.app.video_loading_workflow:
            self.app.video_loading_workflow.save_current_config()

        callback = self._on_dialog_finished_callback
        self._active_wizard = None
        self._on_dialog_finished_callback = None
        if callback is not None:
            callback()
