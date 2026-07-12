"""Review UI for assisted pressed-key exemplar proposals."""

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AssistedCalibrationDecision(str, Enum):
    USE = "use"
    RETRY = "retry"
    KEEP = "keep"


@dataclass
class AssistedCalibrationRow:
    name_label: QLabel
    swatch: QWidget
    status_label: QLabel


class AssistedCalibrationDialog(QDialog):
    SLOT_LABELS = {
        "LW": "Left White",
        "LB": "Left Black",
        "RW": "Right White",
        "RB": "Right Black",
    }

    def __init__(self, proposal, parent=None):
        super().__init__(parent)
        self.proposal = proposal
        self.decision = AssistedCalibrationDecision.KEEP
        self.rows = {}
        self.setWindowTitle(self.tr("Assisted Calibration"))
        self.setModal(True)
        self.setMinimumWidth(480)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.summary_label = QLabel(
            self.tr("{count} samples found across {families} Synthesia note color families.").format(
                count=self.proposal.candidate_count,
                families=self.proposal.assignment_result.family_count,
            )
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.color_family_note = QLabel(
            self.tr("Left/Right refer to Synthesia note colors, not the physical side of the keyboard.")
        )
        self.color_family_note.setWordWrap(True)
        layout.addWidget(self.color_family_note)

        slot_labels = {
            "LW": self.tr("Left White"),
            "LB": self.tr("Left Black"),
            "RW": self.tr("Right White"),
            "RB": self.tr("Right Black"),
        }

        for slot in ("LW", "LB", "RW", "RB"):
            assignment = self.proposal.assignment_result.assignments.get(slot)
            row_layout = QHBoxLayout()
            name_label = QLabel(slot_labels[slot])
            name_label.setMinimumWidth(max(110, name_label.sizeHint().width()))
            row_layout.addWidget(name_label)

            swatch = QWidget()
            swatch.setFixedSize(48, 28)
            swatch.setAccessibleName(self.tr("{name} proposed color").format(name=name_label.text()))
            if assignment is not None and assignment.enabled and assignment.rgb is not None:
                red, green, blue = assignment.rgb
                swatch.setStyleSheet(
                    f"background-color: rgb({red}, {green}, {blue}); border: 1px solid #454545;"
                )
                status = self.tr("Found")
            elif assignment is not None and not assignment.enabled:
                swatch.setStyleSheet("background-color: transparent; border: 1px dashed #595959;")
                status = self.tr("Not used")
            else:
                swatch.setStyleSheet("background-color: transparent; border: 1px dashed #595959;")
                status = self.tr("Not found")
            row_layout.addWidget(swatch)

            status_label = QLabel(status)
            status_label.setMinimumWidth(status_label.sizeHint().width())
            row_layout.addWidget(status_label, 1)
            layout.addLayout(row_layout)
            self.rows[slot] = AssistedCalibrationRow(name_label, swatch, status_label)

        button_layout = QGridLayout()
        self.keep_button = QPushButton(self.tr("Keep Current Examples"))
        self.keep_button.setMinimumHeight(36)
        self.keep_button.clicked.connect(self._keep)
        button_layout.addWidget(self.keep_button, 0, 0)
        self.try_another_button = QPushButton(self.tr("Try Another Frame"))
        self.try_another_button.setMinimumHeight(36)
        self.try_another_button.clicked.connect(self._retry)
        button_layout.addWidget(self.try_another_button, 1, 0)
        self.use_button = QPushButton(self.tr("Use These Examples"))
        self.use_button.setMinimumHeight(36)
        self.use_button.setDefault(True)
        self.use_button.clicked.connect(self._use)
        button_layout.addWidget(self.use_button, 2, 0)
        layout.addLayout(button_layout)
        layout.activate()
        opening_hint = self.sizeHint()
        self.resize(
            max(self.width(), opening_hint.width()),
            max(self.height(), opening_hint.height()),
        )

    def _use(self) -> None:
        self.decision = AssistedCalibrationDecision.USE
        self.accept()

    def _retry(self) -> None:
        self.decision = AssistedCalibrationDecision.RETRY
        self.reject()

    def _keep(self) -> None:
        self.decision = AssistedCalibrationDecision.KEEP
        self.reject()

    def reject(self) -> None:
        if self.decision is not AssistedCalibrationDecision.RETRY:
            self.decision = AssistedCalibrationDecision.KEEP
        super().reject()
