"""Reusable compact editor and review grid for note color families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyle,
    QToolButton,
    QWidget,
)

from synthesia2midi.core.color_families import exemplar_display_parts, slots_for_family


@dataclass
class ExemplarRowWidgets:
    label: QLabel
    swatch: QLabel
    set_button: QPushButton | None = None
    present: QCheckBox | None = None
    status: QLabel | None = None


class ColorFamilyGrid(QWidget):
    """Show two fixed exemplar rows for each visible Synthesia color family."""

    exemplar_requested = Signal(str)
    exemplar_enabled_changed = Signal(str, bool)
    family_add_requested = Signal()
    family_remove_requested = Signal(int)

    def __init__(self, *, mode: Literal["calibration", "review"], parent=None):
        super().__init__(parent)
        if mode not in {"calibration", "review"}:
            raise ValueError(f"Unsupported color family grid mode: {mode}")

        self.mode = mode
        self.rows: dict[str, ExemplarRowWidgets] = {}
        self.remove_family_buttons: dict[int, QToolButton] = {}
        self._family_headings: dict[int, QLabel] = {}
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setHorizontalSpacing(6)
        self._layout.setVerticalSpacing(3)
        self._layout.setColumnMinimumWidth(0, 96)
        self._layout.setColumnMinimumWidth(1, 24)
        self._layout.setColumnMinimumWidth(2, 58)
        self._layout.setColumnStretch(3, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def set_families(
        self,
        family_numbers: Sequence[int],
        *,
        colors: Mapping[str, tuple[int, int, int] | None],
        enabled: Mapping[str, bool],
        assignments: Mapping[str, object] | None = None,
    ) -> None:
        self._rebuild_rows(tuple(family_numbers), colors, enabled, assignments or {})

    def family_heading(self, number: int) -> QLabel:
        return self._family_headings[number]

    def _rebuild_rows(
        self,
        family_numbers: tuple[int, ...],
        colors: Mapping[str, tuple[int, int, int] | None],
        enabled: Mapping[str, bool],
        assignments: Mapping[str, object],
    ) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.rows.clear()
        self.remove_family_buttons.clear()
        self._family_headings.clear()
        if hasattr(self, "add_family_button"):
            del self.add_family_button

        row = 0
        for family_number in family_numbers:
            heading = QLabel(self.tr("Color {number}").format(number=family_number))
            heading.setWordWrap(True)
            heading.setStyleSheet("font-weight: bold;")
            self._layout.addWidget(heading, row, 0, 1, 3)
            self._family_headings[family_number] = heading

            if self.mode == "calibration" and family_number > 1:
                remove_button = self._remove_button(family_number)
                self._layout.addWidget(remove_button, row, 3)
                self.remove_family_buttons[family_number] = remove_button
            row += 1

            for slot in slots_for_family(family_number):
                self.rows[slot] = self._add_exemplar_row(
                    row,
                    slot,
                    colors=colors,
                    enabled=enabled,
                    assignments=assignments,
                )
                row += 1

        if self.mode == "calibration" and len(family_numbers) < 4:
            self.add_family_button = QPushButton(self.tr("Add Color Family"))
            self.add_family_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.add_family_button.clicked.connect(self.family_add_requested)
            self._layout.addWidget(self.add_family_button, row, 0, 1, 4)

        self._layout.invalidate()

    def _add_exemplar_row(
        self,
        row: int,
        slot: str,
        *,
        colors: Mapping[str, tuple[int, int, int] | None],
        enabled: Mapping[str, bool],
        assignments: Mapping[str, object],
    ) -> ExemplarRowWidgets:
        _family_number, display_label = exemplar_display_parts(slot)
        label = QLabel(self.tr(display_label))
        label.setMinimumWidth(96)
        swatch = QLabel()
        swatch.setFixedSize(22, 20)

        assignment = assignments.get(slot)
        assignment_rgb = getattr(assignment, "rgb", None)
        rgb = assignment_rgb if assignment_rgb is not None else colors.get(slot)
        self._set_swatch_color(swatch, rgb)

        self._layout.addWidget(label, row, 0)
        self._layout.addWidget(swatch, row, 1)

        if self.mode == "review":
            status = QLabel(self.tr("Found") if assignment_rgb is not None else self.tr("Missing"))
            status.setMinimumWidth(status.sizeHint().width())
            self._layout.addWidget(status, row, 2, 1, 2)
            return ExemplarRowWidgets(label=label, swatch=swatch, status=status)

        set_button = QPushButton(self.tr("Set"))
        set_button.setMinimumWidth(58)
        set_button.clicked.connect(
            lambda _checked=False, exemplar_slot=slot: self.exemplar_requested.emit(
                exemplar_slot
            )
        )
        present = QCheckBox(self.tr("Present"))
        present.setChecked(enabled.get(slot, False))
        present.toggled.connect(
            lambda is_enabled, exemplar_slot=slot: self.exemplar_enabled_changed.emit(
                exemplar_slot, is_enabled
            )
        )
        self._layout.addWidget(set_button, row, 2)
        self._layout.addWidget(present, row, 3)
        return ExemplarRowWidgets(
            label=label,
            swatch=swatch,
            set_button=set_button,
            present=present,
        )

    def _remove_button(self, family_number: int) -> QToolButton:
        button = QToolButton()
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarCloseButton))
        button.setAutoRaise(True)
        button.setToolTip(self.tr("Remove Color {number}").format(number=family_number))
        button.setAccessibleName(button.toolTip())
        button.clicked.connect(
            lambda _checked=False, number=family_number: self.family_remove_requested.emit(
                number
            )
        )
        return button

    @staticmethod
    def _set_swatch_color(
        swatch: QLabel,
        rgb: tuple[int, int, int] | None,
    ) -> None:
        if rgb is None:
            swatch.setStyleSheet(
                "background-color: transparent; border: 1px dashed #595959;"
            )
            return
        red, green, blue = rgb
        swatch.setStyleSheet(
            f"background-color: rgb({red}, {green}, {blue}); border: 1px solid #454545;"
        )
