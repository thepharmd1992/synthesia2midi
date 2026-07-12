"""Render deterministic offscreen GUI screenshots and report obvious text clipping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSlider,
    QWidget,
)

from synthesia2midi.app_config import OverlayConfig
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.assisted_calibration import (
    AssignedExemplar,
    AssistedCalibrationProposal,
    ExemplarAssignmentResult,
    UnlitFrameAssessment,
)
from synthesia2midi.gui.assisted_calibration_dialog import AssistedCalibrationDialog
from synthesia2midi.gui.auto_detect_tuning_dialog import AutoDetectTuningDialog
from synthesia2midi.gui.calibration_guide import derive_guide_snapshot
from synthesia2midi.gui.controls_qt import ControlPanelQt
from synthesia2midi.gui.manual_keyboard_fit_dialog import ManualKeyboardFitDialog
from synthesia2midi.gui.settings_tool_window import SettingsToolWindow
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

ADVANCED_SECTION_SURFACES = [
    ("settings-advanced-histogram", "histogram"),
    ("settings-advanced-delta", "delta"),
    ("settings-advanced-black-keys", "black_keys"),
    ("settings-advanced-repeated-notes", "repeated_notes"),
    ("settings-advanced-trim", "trim"),
    ("settings-advanced-glossary", "glossary"),
]


def surface_names() -> list[str]:
    return [
        "startup",
        *(name for name, _label in SETTINGS_SURFACES),
        *(name for name, _key in ADVANCED_SECTION_SURFACES),
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


def _settings_surface(
    label: str, *, advanced_section: str | None = None
) -> SettingsToolWindow:
    window = SettingsToolWindow(None)
    panel = ControlPanelQt(window, app_state=AppState())
    window.set_settings_widget(panel)
    labels = [panel.settings_section_rail.item(index).text() for index in range(panel.settings_section_rail.count())]
    source_labels = [source_label for _name, source_label in SETTINGS_SURFACES]
    index = source_labels.index(label)
    # Translation can change the displayed label, but page order remains stable.
    panel.settings_section_rail.setCurrentRow(index)
    if advanced_section is not None:
        panel.advanced_sections[advanced_section]._toggle.setChecked(True)
    window.resize(800, 680)
    return window


def _completed_guide_surface() -> SettingsToolWindow:
    window = _settings_surface("Guide")
    panel = window.settings_widget
    state = panel.app_state
    state.video.filepath = "/tmp/example.mp4"
    state.overlays = [
        OverlayConfig(
            key_id=1,
            note_octave=4,
            note_name_in_octave="C",
            x=0,
            y=0,
            width=10,
            height=40,
            key_type="LW",
            unlit_reference_color=(20, 20, 20),
        )
    ]
    for key_type in state.detection.get_required_base_exemplar_types():
        state.detection.exemplar_lit_colors[key_type] = (220, 40, 30)
    panel.guide_page.update_snapshot(derive_guide_snapshot(state, True))
    return window


def _manual_fit_surface(*, single_overlay: bool) -> ManualKeyboardFitDialog:
    dialog = ManualKeyboardFitDialog()
    if single_overlay:
        dialog.single_overlay_radio.click()
    return dialog


def _youtube_surface(*, populated: bool) -> YouTubeDownloadDialog:
    dialog = YouTubeDownloadDialog(default_output_dir="/tmp")
    if populated:
        url = "https://www.youtube.com/watch?v=example"
        dialog.url_input.setText(url)
        dialog.auto_fetch_timer.stop()
        dialog._on_video_info_fetched(
            url,
            {
                "title": "A Long but Fully Visible Synthesia Piano Tutorial Title",
                "duration": 754,
                "uploader": "Example Piano Channel",
            },
        )
    return dialog


def _repeated_notes_surface() -> QWidget:
    panel = ControlPanelQt(app_state=AppState())
    tool = panel.repeated_notes_tool_window
    tool._matrix_owner = panel
    tool.resize(640, 680)
    return tool


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
    for name, section_key in ADVANCED_SECTION_SURFACES:
        factories.append(
            (
                name,
                lambda key=section_key: _settings_surface("Advanced", advanced_section=key),
            )
        )
    factories.append(("settings-guide-complete", _completed_guide_surface))
    factories.extend(
        [
            ("calibration-wizard", lambda: CalibrationWizard(None, AppState())),
            ("assisted-calibration", lambda: AssistedCalibrationDialog(_assisted_proposal())),
            ("auto-detect-basic", lambda: _auto_detect_surface(False)),
            ("auto-detect-expert", lambda: _auto_detect_surface(True)),
            ("manual-fit", lambda: _manual_fit_surface(single_overlay=False)),
            ("manual-fit-single", lambda: _manual_fit_surface(single_overlay=True)),
            ("youtube-download", lambda: _youtube_surface(populated=False)),
            ("youtube-download-populated", lambda: _youtube_surface(populated=True)),
            ("repeated-notes-tool", _repeated_notes_surface),
        ]
    )
    return factories


def _detect_clipping(widget: QWidget) -> list[str]:
    def control_text(control: QWidget) -> str:
        if isinstance(control, QGroupBox):
            return control.title()
        if hasattr(control, "text"):
            return control.text()
        return control.metaObject().className()

    findings = []
    for button in widget.findChildren(QAbstractButton):
        if not button.isVisible() or not button.text().strip():
            continue
        if button.text().strip() in {"+", "-", "0"}:
            continue
        required = button.sizeHint().width()
        if button.width() < required or button.height() < button.sizeHint().height():
            findings.append(f"{button.metaObject().className()}:{button.text()}")
    for label in widget.findChildren(QLabel):
        if not label.isVisible() or not label.text().strip() or "\n" in label.text():
            continue
        required = 0 if label.wordWrap() else label.sizeHint().width()
        required_height = (
            label.heightForWidth(label.width())
            if label.wordWrap() and label.hasHeightForWidth()
            else label.sizeHint().height()
        )
        if label.width() + 2 < required or label.height() + 2 < required_height:
            findings.append(f"QLabel:{label.text()}")

    overlap_types = (
        QAbstractButton,
        QAbstractSpinBox,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QSlider,
    )
    candidates = [
        child
        for child in widget.findChildren(QWidget)
        if child.isVisible() and isinstance(child, overlap_types)
    ]
    for child in candidates:
        ancestor = child.parentWidget()
        scroll_area = None
        while ancestor is not None and ancestor is not widget:
            if isinstance(ancestor, QAbstractScrollArea):
                scroll_area = ancestor
                break
            ancestor = ancestor.parentWidget()

        visible_rect = child.visibleRegion().boundingRect()
        if visible_rect.isEmpty():
            continue
        width_clipped = visible_rect.width() + 2 < child.width()
        height_clipped = visible_rect.height() + 2 < child.height()
        if scroll_area is not None:
            width_clipped = (
                width_clipped
                and scroll_area.horizontalScrollBar().maximum() == 0
            )
            height_clipped = (
                height_clipped
                and scroll_area.verticalScrollBar().maximum() == 0
            )
        if width_clipped or height_clipped:
            findings.append(f"ancestor-clip:{control_text(child)}")

    parents = {child.parentWidget() for child in candidates}
    for parent in parents:
        siblings = [child for child in candidates if child.parentWidget() is parent]
        for index, child in enumerate(siblings):
            for other in siblings[index + 1 :]:
                if not child.geometry().intersects(other.geometry()):
                    continue
                findings.append(
                    f"overlap:{control_text(child)}<>{control_text(other)}"
                )
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


def _grab_logical_image(widget: QWidget):
    """Normalize high-DPI captures to the widget's logical pixel dimensions."""
    image = widget.grab().toImage()
    if image.size() != widget.size():
        image = image.scaled(
            widget.size(),
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation,
        )
    image.setDevicePixelRatio(1.0)
    return image


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
            widget.show()
            app.processEvents()
            image = _grab_logical_image(widget)
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
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
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
