"""Review UI for assisted pressed-key exemplar proposals."""

from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
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

        for slot in ("LW", "LB", "RW", "RB"):
            assignment = self.proposal.assignment_result.assignments.get(slot)
            row_layout = QHBoxLayout()
            name_label = QLabel(self.tr(self.SLOT_LABELS[slot]))
            name_label.setMinimumWidth(110)
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
            row_layout.addWidget(status_label, 1)
            layout.addLayout(row_layout)
            self.rows[slot] = AssistedCalibrationRow(name_label, swatch, status_label)

        button_layout = QHBoxLayout()
        self.keep_button = QPushButton(self.tr("Keep Current Examples"))
        self.keep_button.clicked.connect(self._keep)
        button_layout.addWidget(self.keep_button)
        self.try_another_button = QPushButton(self.tr("Try Another Frame"))
        self.try_another_button.clicked.connect(self._retry)
        button_layout.addWidget(self.try_another_button)
        button_layout.addStretch(1)
        self.use_button = QPushButton(self.tr("Use These Examples"))
        self.use_button.setDefault(True)
        self.use_button.clicked.connect(self._use)
        button_layout.addWidget(self.use_button)
        layout.addLayout(button_layout)

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
