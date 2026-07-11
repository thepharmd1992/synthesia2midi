"""Render deterministic offscreen GUI screenshots and report obvious text clipping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget

from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDialog
from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.manual_keyboard_fit_dialog import ManualKeyboardFitDialog
from synthesia2midi.gui.startup_dialog import StartupDialog
from synthesia2midi.gui.wizard import CalibrationWizard
from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.localization import install_translator


SETTINGS_SURFACES = [
    ("settings-guide", "Guide"),
    ("settings-calibration", "Calibration"),
    ("settings-overlays", "Overlays"),
    ("settings-detection", "Detection"),
    ("settings-midi", "MIDI"),
    ("settings-advanced", "Advanced"),
    ("settings-optional", "Optional"),
    ("settings-language", "Language"),
]


def surface_names() -> list[str]:
    return [
        "startup",
        *(name for name, _label in SETTINGS_SURFACES),
        "calibration-wizard",
        "assisted-calibration",
        "auto-detect-basic",
        "auto-detect-expert",
        "manual-fit",
        "youtube-download",
    ]


def _assisted_proposal() -> AssistedCalibrationProposal:
    assignments = {}
    colors = {"LW": (220, 40, 30), "LB": (150, 25, 20), "RW": (20, 100, 240), "RB": None}
    for slot, color in colors.items():
        assignments[slot] = AssignedExemplar(
            slot=slot,
            rgb=color,
            hist=np.ones(4, dtype=np.float32) if color is not None else None,
            source=None,
            enabled=True,
        )
    return AssistedCalibrationProposal(
        baseline_frame_index=0,
        unlit_assessment=UnlitFrameAssessment(status="clean"),
        assignment_result=ExemplarAssignmentResult(
            assignments=assignments,
            missing_slots=("RB",),
            disabled_slots=(),
            family_count=2,
            confidence=0.9,
        ),
        scanned_frame_count=120,
        candidate_count=12,
    )


def _settings_surface(label: str) -> ControlPanelQt:
    panel = ControlPanelQt(app_state=AppState())
    labels = [panel.settings_section_rail.item(index).text() for index in range(panel.settings_section_rail.count())]
    source_labels = [source_label for _name, source_label in SETTINGS_SURFACES]
    index = source_labels.index(label)
    # Translation can change the displayed label, but page order remains stable.
    panel.settings_section_rail.setCurrentRow(index)
    panel.resize(max(820, panel.sizeHint().width()), 900)
    return panel


def _auto_detect_surface(expert: bool) -> AutoDetectTuningDialog:
    dialog = AutoDetectTuningDialog(
        None,
        AppState(),
        np.zeros((16, 32, 3), dtype=np.uint8),
        (0, 0, 32, 16),
        initial_detection_results={"total_keys": 88, "white_keys": 52, "black_keys": 36},
        fallback_used=False,
        apply_detection_callback=lambda _result: True,
    )
    dialog.tabs.setCurrentIndex(1 if expert else 0)
    return dialog


def _surface_factories() -> list[tuple[str, Callable[[], QWidget]]]:
    factories: list[tuple[str, Callable[[], QWidget]]] = [
        ("startup", lambda: StartupDialog(recent_video_paths=[])),
    ]
    for name, label in SETTINGS_SURFACES:
        factories.append((name, lambda page_label=label: _settings_surface(page_label)))
    factories.extend(
        [
            ("calibration-wizard", lambda: CalibrationWizard(None, AppState())),
            ("assisted-calibration", lambda: AssistedCalibrationDialog(_assisted_proposal())),
            ("auto-detect-basic", lambda: _auto_detect_surface(False)),
            ("auto-detect-expert", lambda: _auto_detect_surface(True)),
            ("manual-fit", lambda: ManualKeyboardFitDialog()),
            ("youtube-download", lambda: YouTubeDownloadDialog(default_output_dir="/tmp")),
        ]
    )
    return factories


def _detect_clipping(widget: QWidget) -> list[str]:
    findings = []
    for button in widget.findChildren(QAbstractButton):
        if not button.isVisible() or not button.text().strip():
            continue
        if button.text().strip() in {"+", "-", "0"}:
            continue
        required = button.sizeHint().width()
        if button.width() < required:
            findings.append(f"{button.metaObject().className()}:{button.text()}")
    for label in widget.findChildren(QLabel):
        if not label.isVisible() or label.wordWrap() or not label.text().strip() or "\n" in label.text():
            continue
        required = label.sizeHint().width()
        if label.width() + 2 < required:
            findings.append(f"QLabel:{label.text()}")
    return sorted(set(findings))


def _image_nonblank(image) -> bool:
    if image.isNull() or image.width() < 2 or image.height() < 2:
        return False
    colors = set()
    x_step = max(1, image.width() // 80)
    y_step = max(1, image.height() // 80)
    for y in range(0, image.height(), y_step):
        for x in range(0, image.width(), x_step):
            colors.add(image.pixelColor(x, y).rgba())
            if len(colors) > 1:
                return True
    return False


def render_matrix(output: Path, *, locale_name: str, font_scale: float) -> int:
    app = QApplication.instance() or QApplication([])
    original_font = QFont(app.font())
    scaled_font = QFont(original_font)
    base_size = original_font.pointSizeF() if original_font.pointSizeF() > 0 else 13.0
    scaled_font.setPointSizeF(base_size * font_scale)
    output.mkdir(parents=True, exist_ok=True)
    install_translator(app, locale_name)
    app.setFont(scaled_font)

    report_entries = []
    try:
        for name, factory in _surface_factories():
            widget = factory()
            widget.resize(widget.sizeHint().expandedTo(widget.minimumSizeHint()))
            widget.show()
            app.processEvents()
            image = widget.grab().toImage()
            image.save(str(output / f"{name}.png"))
            report_entries.append(
                {
                    "surface": name,
                    "width": image.width(),
                    "height": image.height(),
                    "nonblank": _image_nonblank(image),
                    "clipping": _detect_clipping(widget),
                }
            )
            widget.close()
            widget.deleteLater()
            app.processEvents()
    finally:
        install_translator(app, "en")
        app.setFont(original_font)

    report = {
        "schema_version": 1,
        "locale": locale_name,
        "font_scale": font_scale,
        "surfaces": report_entries,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if any(not item["nonblank"] or item["clipping"] for item in report_entries) else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale", default="qps")
    parser.add_argument("--font-scale", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return render_matrix(args.output, locale_name=args.locale, font_scale=args.font_scale)


if __name__ == "__main__":
    raise SystemExit(main())
