import json

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from synthesia2midi.gui.settings_tool_window import SettingsToolWindow
from synthesia2midi.tools import render_ui_matrix


def test_ui_matrix_registers_all_required_surfaces():
    names = render_ui_matrix.surface_names()
    assert names == [
        "startup",
        "settings-guide",
        "settings-calibration",
        "settings-overlays",
        "settings-detection",
        "settings-midi",
        "settings-advanced",
        "settings-optional",
        "settings-language",
        "settings-advanced-histogram",
        "settings-advanced-delta",
        "settings-advanced-black-keys",
        "settings-advanced-repeated-notes",
        "settings-advanced-trim",
        "settings-advanced-glossary",
        "settings-guide-complete",
        "calibration-wizard",
        "assisted-calibration",
        "auto-detect-basic",
        "auto-detect-expert",
        "manual-fit",
        "manual-fit-single",
        "youtube-download",
        "youtube-download-populated",
        "repeated-notes-tool",
    ]


def test_settings_matrix_surface_uses_real_production_window():
    QApplication.instance() or QApplication([])
    window = render_ui_matrix._settings_surface("Language")
    try:
        assert isinstance(window, SettingsToolWindow)
        assert window.size() == QSize(780, 600)
        panel = window.settings_widget
        assert panel is not None
        assert panel.settings_section_rail.currentRow() == 7
        assert panel.settings_page_scroll_areas[7].verticalScrollBar().maximum() == 0
    finally:
        window.close()


def test_ui_matrix_writes_nonblank_screenshots_and_stable_report(tmp_path):
    QApplication.instance() or QApplication([])
    exit_code = render_ui_matrix.render_matrix(tmp_path, locale_name="qps", font_scale=1.5)
    report = json.loads((tmp_path / "report.json").read_text())

    failures = {
        entry["surface"]: entry["clipping"]
        for entry in report["surfaces"]
        if not entry["nonblank"] or entry["clipping"]
    }
    assert exit_code == 0, failures
    assert [entry["surface"] for entry in report["surfaces"]] == render_ui_matrix.surface_names()
    assert all(entry["nonblank"] for entry in report["surfaces"])
    assert all(entry["clipping"] == [] for entry in report["surfaces"])
    assert all(entry["width"] <= 800 and entry["height"] <= 720 for entry in report["surfaces"])
    by_surface = {entry["surface"]: entry for entry in report["surfaces"]}
    assert by_surface["manual-fit-single"]["height"] < by_surface["manual-fit"]["height"]
    assert all((tmp_path / f"{name}.png").is_file() for name in render_ui_matrix.surface_names())


def test_ui_matrix_returns_nonzero_when_clipping_is_reported(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])
    monkeypatch.setattr(render_ui_matrix, "_detect_clipping", lambda _widget: ["forced"])

    assert render_ui_matrix.render_matrix(tmp_path, locale_name="en", font_scale=1.0) == 1


def test_ui_matrix_preserves_surface_opening_geometry(monkeypatch, tmp_path):
    QApplication.instance() or QApplication([])

    class OpeningSizeWidget(QWidget):
        def sizeHint(self):
            return QSize(700, 500)

        def grab(self):
            pixmap = QPixmap(self.width() * 2, self.height() * 2)
            pixmap.setDevicePixelRatio(2.0)
            pixmap.fill(Qt.white)
            return pixmap

    widget = OpeningSizeWidget()
    widget.resize(321, 123)
    monkeypatch.setattr(render_ui_matrix, "_surface_factories", lambda: [("opening", lambda: widget)])

    render_ui_matrix.render_matrix(tmp_path, locale_name="en", font_scale=1.0)
    report = json.loads((tmp_path / "report.json").read_text())

    assert report["surfaces"][0]["width"] == 321
    assert report["surfaces"][0]["height"] == 123
    assert QImage(str(tmp_path / "opening.png")).size() == QSize(321, 123)


def test_ui_matrix_reports_overlapping_sibling_controls():
    QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(240, 100)
    label = QLabel("Explanation", parent)
    label.setGeometry(10, 10, 140, 40)
    button = QPushButton("Choice", parent)
    button.setGeometry(10, 30, 140, 40)
    parent.show()
    QApplication.processEvents()
    try:
        assert any(
            finding.startswith("overlap:")
            for finding in render_ui_matrix._detect_clipping(parent)
        )
    finally:
        parent.close()


def test_ui_matrix_reports_visible_control_clipped_by_non_scroll_ancestor():
    QApplication.instance() or QApplication([])
    parent = QWidget()
    parent.resize(240, 100)
    container = QWidget(parent)
    container.setGeometry(0, 0, 240, 50)
    label = QLabel("Partially visible", container)
    label.setGeometry(10, 30, 180, 40)
    parent.show()
    QApplication.processEvents()
    try:
        assert any(
            finding.startswith("ancestor-clip:")
            for finding in render_ui_matrix._detect_clipping(parent)
        )
    finally:
        parent.close()
