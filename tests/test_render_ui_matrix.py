import json

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

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
        "calibration-wizard",
        "assisted-calibration",
        "auto-detect-basic",
        "auto-detect-expert",
        "manual-fit",
        "youtube-download",
    ]


def test_ui_matrix_writes_nonblank_screenshots_and_stable_report(tmp_path):
    QApplication.instance() or QApplication([])
    exit_code = render_ui_matrix.render_matrix(tmp_path, locale_name="qps", font_scale=1.5)
    report = json.loads((tmp_path / "report.json").read_text())

    assert exit_code == 0
    assert [entry["surface"] for entry in report["surfaces"]] == render_ui_matrix.surface_names()
    assert all(entry["nonblank"] for entry in report["surfaces"])
    assert all(entry["clipping"] == [] for entry in report["surfaces"])
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

    widget = OpeningSizeWidget()
    widget.resize(321, 123)
    monkeypatch.setattr(render_ui_matrix, "_surface_factories", lambda: [("opening", lambda: widget)])

    render_ui_matrix.render_matrix(tmp_path, locale_name="en", font_scale=1.0)
    report = json.loads((tmp_path / "report.json").read_text())

    assert report["surfaces"][0]["width"] == 321
    assert report["surfaces"][0]["height"] == 123


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
