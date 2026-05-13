"""Facade for calibration interaction controllers used by the main window."""
from __future__ import annotations

from typing import Optional

import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.gui.overlay_interaction_controller import OverlayInteractionController
from synthesia2midi.gui.shadow_calibration_controller import ShadowCalibrationController
from synthesia2midi.gui.spark_calibration_controller import SparkCalibrationController


class CalibrationEffectsController:
    """Compatibility facade over focused spark, shadow, and overlay controllers."""

    def __init__(self, app):
        self.spark = SparkCalibrationController(app)
        self.shadow = ShadowCalibrationController(app)
        self.overlay = OverlayInteractionController(app)

    def _handle_spark_roi_selection_request(self):
        return self.spark._handle_spark_roi_selection_request()

    def _handle_spark_roi_visibility_toggle(self, visible: bool):
        return self.spark._handle_spark_roi_visibility_toggle(visible)

    def _handle_spark_roi_updated(self, top_y: int, bottom_y: int):
        return self.spark._handle_spark_roi_updated(top_y, bottom_y)

    def _handle_spark_calibration_request(self, step_type: str):
        return self.spark._handle_spark_calibration_request(step_type)

    def _handle_auto_spark_calibration_request(self, key_type: str):
        return self.spark._handle_auto_spark_calibration_request(key_type)

    def _handle_spark_detection_toggle(self, enabled: bool):
        return self.spark._handle_spark_detection_toggle(enabled)

    def _handle_spark_detection_sensitivity_change(self, value: float):
        return self.spark._handle_spark_detection_sensitivity_change(value)

    def _capture_spark_background_calibration(self):
        return self.spark._capture_spark_background_calibration()

    def _capture_spark_overlay_calibration(self, overlay, calibration_mode: str):
        return self.spark._capture_spark_overlay_calibration(overlay, calibration_mode)

    def _get_calibration_instructions(self, step_type: str) -> str:
        return self.spark._get_calibration_instructions(step_type)

    def _handle_shadow_roi_selection_request(self):
        return self.shadow._handle_shadow_roi_selection_request()

    def _handle_shadow_white_roi_selection_request(self):
        return self.shadow._handle_shadow_white_roi_selection_request()

    def _handle_shadow_black_roi_selection_request(self):
        return self.shadow._handle_shadow_black_roi_selection_request()

    def _handle_shadow_detection_toggle(self, enabled: bool):
        return self.shadow._handle_shadow_detection_toggle(enabled)

    def _handle_shadow_detection_sensitivity_change(self, value: float):
        return self.shadow._handle_shadow_detection_sensitivity_change(value)

    def _handle_shadow_darkness_threshold_change(self, value: float):
        return self.shadow._handle_shadow_darkness_threshold_change(value)

    def _handle_shadow_calibration_request(self, key_type: str, calibration_type: str):
        return self.shadow._handle_shadow_calibration_request(key_type, calibration_type)

    def _capture_shadow_overlay_calibration(self, overlay, calibration_mode: str):
        return self.shadow._capture_shadow_overlay_calibration(overlay, calibration_mode)

    def _extract_roi(self, frame: np.ndarray, overlay: OverlayConfig) -> Optional[np.ndarray]:
        return self.shadow._extract_roi(frame, overlay)

    def _handle_overlay_type_change(self, overlay_type: str):
        return self.overlay._handle_overlay_type_change(overlay_type)
