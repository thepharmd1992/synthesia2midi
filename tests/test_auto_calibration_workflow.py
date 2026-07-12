from types import SimpleNamespace

import cv2
import numpy as np

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.gui.spark_calibration_controller import SparkCalibrationController
from synthesia2midi.workflows.auto_calibration import AutoCalibrationWorkflow


def _overlay(*, key_type="RW"):
    return OverlayConfig(
        key_id=1,
        note_octave=4,
        note_name_in_octave="C",
        x=0,
        y=0,
        width=4,
        height=4,
        key_type=key_type,
    )


def test_overlay_click_preserves_selected_canonical_spark_slot(monkeypatch):
    state = AppState()
    state.video.current_frame_index = 42
    state.calibration.current_calibration_key_type = "RW"
    workflow = AutoCalibrationWorkflow(state, SimpleNamespace())
    workflow.calibration_in_progress = True
    monkeypatch.setattr(workflow, "_execute_auto_calibration", lambda: True)

    left_hued_hsv = np.full((4, 4, 3), (80, 255, 255), dtype=np.uint8)
    left_hued_bgr = cv2.cvtColor(left_hued_hsv, cv2.COLOR_HSV2BGR)

    assert workflow.handle_overlay_click(_overlay(), left_hued_bgr) is True
    assert workflow.current_request is not None
    assert workflow.current_request.requested_key_type == "RW"


def test_spark_auto_calibration_popup_uses_color_family_label(monkeypatch):
    messages = []
    app = SimpleNamespace(
        auto_calibration_workflow=SimpleNamespace(
            start_auto_calibration=lambda slot: slot == "RW"
        )
    )
    monkeypatch.setattr(
        "synthesia2midi.gui.spark_calibration_controller.QMessageBox.information",
        lambda parent, title, message: messages.append((title, message)),
    )

    SparkCalibrationController(app).start_auto_spark_calibration("RW")

    assert messages[0][0] == "Auto-Calibrate Color 2 Natural"
    assert "Auto-calibration for Color 2 Natural started." in messages[0][1]
    assert "RW" not in messages[0][1]
    assert "left/right hand" not in messages[0][1]
    assert "Key Type Legend" not in messages[0][1]
