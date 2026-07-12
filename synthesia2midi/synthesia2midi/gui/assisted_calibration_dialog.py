"""Review UI for assisted pressed-key exemplar proposals."""

from enum import Enum

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from synthesia2midi.core.color_families import active_family_numbers
from synthesia2midi.gui.color_family_grid import ColorFamilyGrid


class AssistedCalibrationDecision(str, Enum):
    USE = "use"
    RETRY = "retry"
    KEEP = "keep"


class AssistedCalibrationDialog(QDialog):
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

        self.warning_banner = QLabel("\n".join(self.proposal.warnings))
        self.warning_banner.setWordWrap(True)
        self.warning_banner.setStyleSheet(
            "background-color: #fff4ce; border: 1px solid #d9a400; color: #5c3b00; padding: 6px;"
        )
        self.warning_banner.setVisible(bool(self.proposal.warnings))
        layout.addWidget(self.warning_banner)

        assignments = self.proposal.assignment_result.assignments
        colors = {
            slot: assignment.rgb for slot, assignment in assignments.items()
        }
        enabled = {
            slot: assignment.enabled for slot, assignment in assignments.items()
        }
        self.color_family_grid = ColorFamilyGrid(mode="review")
        self.color_family_grid.set_families(
            active_family_numbers(enabled, colors),
            colors=colors,
            enabled=enabled,
            assignments=assignments,
        )
        layout.addWidget(self.color_family_grid)
        self.rows = self.color_family_grid.rows

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
