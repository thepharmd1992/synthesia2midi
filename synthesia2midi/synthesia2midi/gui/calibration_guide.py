"""Beginner-facing calibration progress derived from existing app state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QCoreApplication, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from synthesia2midi.core.app_state import AppState


translate = QCoreApplication.translate


class GuideStatus(str, Enum):
    DONE = "done"
    NEXT = "next"
    NEEDS_REVIEW = "needs_review"
    NOT_READY = "not_ready"


@dataclass(frozen=True)
class GuideStepState:
    key: str
    status: GuideStatus


@dataclass(frozen=True)
class GuideSnapshot:
    video: GuideStepState
    overlays: GuideStepState
    unlit: GuideStepState
    exemplars: GuideStepState
    conversion: GuideStepState

    @property
    def steps(self) -> tuple[GuideStepState, ...]:
        return (self.video, self.overlays, self.unlit, self.exemplars, self.conversion)


def derive_guide_snapshot(app_state: AppState, conversion_ready: bool) -> GuideSnapshot:
    """Derive beginner progress without adding persisted workflow state."""
    has_video = bool(getattr(app_state.video, "filepath", ""))
    overlays = tuple(getattr(app_state, "overlays", ()) or ())
    has_overlays = bool(overlays)
    histogram_required = bool(app_state.detection.use_histogram_detection)
    has_unlit = has_overlays and all(
        overlay.unlit_reference_color is not None
        and (not histogram_required or overlay.unlit_hist is not None)
        for overlay in overlays
    )
    required_exemplars = app_state.detection.get_required_base_exemplar_types()
    effective_colors = app_state.detection.get_effective_exemplar_lit_colors()
    has_exemplars = bool(required_exemplars) and all(
        effective_colors.get(key_type) is not None for key_type in required_exemplars
    )
    has_any_downstream_calibration = any(
        overlay.unlit_reference_color is not None or overlay.unlit_hist is not None
        for overlay in overlays
    ) or any(color is not None for color in effective_colors.values())

    video = GuideStepState("video", GuideStatus.DONE if has_video else GuideStatus.NEXT)
    if not has_video:
        overlays_status = GuideStatus.NOT_READY
    elif not has_overlays:
        overlays_status = GuideStatus.NEXT
    elif has_any_downstream_calibration:
        overlays_status = GuideStatus.DONE
    else:
        overlays_status = GuideStatus.NEEDS_REVIEW

    if not has_overlays or overlays_status is GuideStatus.NEEDS_REVIEW:
        unlit_status = GuideStatus.NOT_READY
    elif has_unlit:
        unlit_status = GuideStatus.DONE
    else:
        unlit_status = GuideStatus.NEXT

    if not has_unlit:
        exemplar_status = GuideStatus.NOT_READY
    elif has_exemplars:
        exemplar_status = GuideStatus.DONE
    else:
        exemplar_status = GuideStatus.NEXT

    return GuideSnapshot(
        video,
        GuideStepState("overlays", overlays_status),
        GuideStepState("unlit", unlit_status),
        GuideStepState("exemplars", exemplar_status),
        GuideStepState(
            "conversion",
            GuideStatus.NEXT if conversion_ready else GuideStatus.NOT_READY,
        ),
    )


class KeyboardExample(QWidget):
    """Small deterministic keyboard illustration used by the Guide."""

    def __init__(self, *, glowing_key: bool = False, boxed: bool = False, parent=None):
        super().__init__(parent)
        self.glowing_key = glowing_key
        self.boxed = boxed
        self.setFixedHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAccessibleName(translate("CalibrationGuideWidget", "Keyboard calibration example"))

    def paintEvent(self, event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        bounds = QRectF(4, 4, max(1, self.width() - 8), self.height() - 8)
        key_width = bounds.width() / 10
        for index in range(10):
            color = QColor("#30a7ff") if self.glowing_key and index == 6 else QColor("#ffffff")
            rect = QRectF(bounds.left() + index * key_width, bounds.top(), key_width, bounds.height())
            painter.fillRect(rect, color)
            painter.setPen(QPen(QColor("#454545"), 1))
            painter.drawRect(rect)
        if self.boxed:
            painter.setPen(QPen(QColor("#d32f2f"), 3))
            painter.drawRect(bounds.adjusted(1, 1, -1, -1))


class GuideStepRow(QWidget):
    def __init__(self, title: str, instruction: str, action_text: str, illustration=None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(5)

        heading = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600;")
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        heading.addWidget(self.title_label, 1)
        heading.addWidget(self.status_label)
        layout.addLayout(heading)

        self.instruction_label = QLabel(instruction)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setMinimumHeight(self.instruction_label.sizeHint().height())
        layout.addWidget(self.instruction_label)
        if illustration is not None:
            layout.addWidget(illustration)

        self.primary_button = QPushButton(action_text)
        self.primary_button.setMinimumHeight(36)
        layout.addWidget(self.primary_button, 0, Qt.AlignLeft)

    def set_status(self, status: GuideStatus) -> None:
        labels = {
            GuideStatus.DONE: translate("CalibrationGuideWidget", "Done"),
            GuideStatus.NEXT: translate("CalibrationGuideWidget", "Next"),
            GuideStatus.NEEDS_REVIEW: translate("CalibrationGuideWidget", "Needs review"),
            GuideStatus.NOT_READY: translate("CalibrationGuideWidget", "Not ready"),
        }
        self.status_label.setText(labels[status])
        self.status_label.setProperty("guideStatus", status.value)


class CalibrationGuideWidget(QWidget):
    open_video_requested = Signal()
    youtube_requested = Signal()
    find_keyboard_requested = Signal()
    review_alignment_requested = Signal()
    capture_unlit_requested = Signal()
    assisted_scan_requested = Signal()
    convert_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 8, 8)
        layout.setSpacing(4)

        title = QLabel(self.tr("Start here"))
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(title)
        intro = QLabel(self.tr("Follow the next highlighted step. Your existing calibration is detected automatically."))
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.assisted_status_label = QLabel()
        self.assisted_status_label.setWordWrap(True)
        self.assisted_status_label.hide()
        layout.addWidget(self.assisted_status_label)

        definitions = [
            (self.tr("1. Open or download a video"), self.tr("Use a clear Synthesia-style piano video with visible keys and falling notes."), self.tr("Open Video"), None, self.open_video_requested),
            (self.tr("2. Find and check the keyboard overlays"), self.tr("Draw around the full keyboard, then check that the boxes line up with the keys."), self.tr("Find Keyboard"), KeyboardExample(boxed=True), None),
            (self.tr("3. Capture a no-key frame"), self.tr("Pause where the keyboard is visible and no keys are glowing."), self.tr("Capture No-Key Frame"), KeyboardExample(), self.capture_unlit_requested),
            (self.tr("4. Find pressed-key colors"), self.tr("Start from the no-key frame. The scan looks ahead for each enabled Left/Right color family."), self.tr("Find Pressed-Key Colors"), KeyboardExample(glowing_key=True), self.assisted_scan_requested),
            (self.tr("5. Create MIDI"), self.tr("When every required step is done, create the MIDI file."), self.tr("Create MIDI"), None, self.convert_requested),
        ]
        self.step_rows = []
        for title_text, instruction, action, illustration, signal in definitions:
            row = GuideStepRow(title_text, instruction, action, illustration)
            if signal is not None:
                row.primary_button.clicked.connect(signal.emit)
            layout.addWidget(row)
            self.step_rows.append(row)
        self._review_existing_overlays = False
        self.step_rows[1].primary_button.clicked.connect(self._handle_overlay_action)
        self.youtube_button = QPushButton(self.tr("Download from YouTube"))
        self.youtube_button.setMinimumHeight(36)
        self.youtube_button.clicked.connect(self.youtube_requested.emit)
        self.step_rows[0].layout().addWidget(self.youtube_button, 0, Qt.AlignLeft)
        layout.addStretch(1)

    def _handle_overlay_action(self) -> None:
        if self._review_existing_overlays:
            self.review_alignment_requested.emit()
        else:
            self.find_keyboard_requested.emit()

    def update_snapshot(self, snapshot: GuideSnapshot) -> None:
        for row, step in zip(self.step_rows, snapshot.steps):
            row.set_status(step.status)
            row.primary_button.setEnabled(step.status is not GuideStatus.NOT_READY)
        self._review_existing_overlays = snapshot.overlays.status is GuideStatus.NEEDS_REVIEW
        self.step_rows[1].primary_button.setText(
            self.tr("Review Alignment")
            if self._review_existing_overlays
            else self.tr("Find Keyboard")
        )

    def set_assisted_state(self, state: str) -> None:
        messages = {
            "scanning": self.tr("Scanning the video for pressed-key colors..."),
            "applied": self.tr("Pressed-key colors updated."),
            "none_found": self.tr("No pressed-key colors were found. Move to another no-key frame and try again."),
            "retry": self.tr("Move to another no-key frame, then run the scan again."),
            "kept": self.tr("Your current pressed-key colors were kept."),
        }
        message = messages.get(state, "")
        self.assisted_status_label.setText(message)
        self.assisted_status_label.setVisible(bool(message))
