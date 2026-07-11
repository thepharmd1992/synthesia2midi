import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget

from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.auto_detect_param_specs import (
    get_advanced_auto_detect_param_keys,
    get_basic_auto_detect_param_keys,
)
from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog


def test_auto_detect_basic_params_only_include_edge_drift_controls():
    assert get_basic_auto_detect_param_keys() == [
        "white_edge_left_shift_ticks",
        "white_edge_right_shift_ticks",
    ]

    advanced_keys = set(get_advanced_auto_detect_param_keys())
    assert {
        "black_upper_ratio",
        "black_bottom_ratio",
        "white_bottom_ratio",
        "white_initial_top_ratio",
        "padding_percent",
    }.issubset(advanced_keys)


def test_auto_detect_tuning_dialog_removes_default_profile_and_geometry_copy():
    QApplication.instance() or QApplication([])
    dialog = AutoDetectTuningDialog(
        None,
        AppState(),
        np.zeros((8, 8, 3), dtype=np.uint8),
        (0, 0, 8, 8),
        initial_detection_results={"total_keys": 88},
        fallback_used=False,
        apply_detection_callback=lambda _results: True,
    )

    try:
        label_texts = [label.text() for label in dialog.findChildren(QLabel)]

        assert "Initial auto-detect used the default built-in profile." not in label_texts
        assert "White-from-black geometry mode is always enabled." not in label_texts
        assert dialog.size().height() <= 520
        assert dialog.size().width() <= 800
    finally:
        dialog.close()


def test_auto_detect_tuning_dialog_uses_user_guidance_copy():
    QApplication.instance() or QApplication([])
    dialog = AutoDetectTuningDialog(
        None,
        AppState(),
        np.zeros((8, 8, 3), dtype=np.uint8),
        (0, 0, 8, 8),
        initial_detection_results={"total_keys": 88},
        fallback_used=False,
        apply_detection_callback=lambda _results: True,
    )

    try:
        label_texts = [label.text() for label in dialog.findChildren(QLabel)]
        button_texts = [button.text() for button in dialog.findChildren(QPushButton)]
        tabs = dialog.findChild(QTabWidget)

        assert (
            "Check the overlays on the video. If they line up with the keys, click Save. "
            "If the edges are off, adjust the edge controls."
        ) in label_texts
        assert "Reset to Recommended Settings" in button_texts
        assert tabs.tabText(1) == "Advanced (Expert)"
        assert tabs.currentIndex() == 0
        assert dialog.expert_note.text() == (
            "Use these controls only when Basic edge alignment cannot line the overlays up with the keys."
        )
        assert dialog.expert_sections
        assert all(not section._content.isVisible() for section in dialog.expert_sections)
    finally:
        dialog.close()
