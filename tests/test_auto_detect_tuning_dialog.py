import numpy as np
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget
from PySide6.QtCore import Qt

from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.auto_detect_param_specs import (
    get_advanced_auto_detect_param_keys,
    get_basic_auto_detect_param_keys,
)
from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog
from synthesia2midi.localization import install_translator


def _make_dialog():
    return AutoDetectTuningDialog(
        None,
        AppState(),
        np.zeros((8, 8, 3), dtype=np.uint8),
        (0, 0, 8, 8),
        initial_detection_results={"total_keys": 88},
        fallback_used=False,
        apply_detection_callback=lambda _results: True,
    )


def _assert_basic_edge_controls_fit(dialog):
    dialog.show()
    QApplication.processEvents()
    reset_button = next(
        button
        for button in dialog.basic_scroll_area.findChildren(QPushButton)
        if button.text()
        == QCoreApplication.translate("AutoDetectTuningDialog", "Reset Section")
    )

    assert dialog.basic_scroll_area.verticalScrollBar().maximum() == 0
    assert reset_button.isVisible()


def test_basic_edge_controls_fit_without_scrolling_at_default_font():
    QApplication.instance() or QApplication([])
    dialog = _make_dialog()
    try:
        _assert_basic_edge_controls_fit(dialog)
    finally:
        dialog.close()


def test_basic_edge_controls_fit_without_scrolling_in_large_pseudo_locale():
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    install_translator(app, "qps")
    scaled_font = QFont(original_font)
    scaled_font.setPointSizeF(original_font.pointSizeF() * 1.5)
    app.setFont(scaled_font)
    dialog = _make_dialog()
    try:
        _assert_basic_edge_controls_fit(dialog)
    finally:
        dialog.close()
        install_translator(app, "en")
        app.setFont(original_font)


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


def test_return_activates_save_instead_of_reset():
    QApplication.instance() or QApplication([])
    dialog = _make_dialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog.show()
    QApplication.processEvents()
    slider = dialog._control_widgets["white_edge_left_shift_ticks"]["slider"]
    slider.setFocus()

    QTest.keyClick(slider, Qt.Key_Return)
    QApplication.processEvents()

    assert accepted == [True]
