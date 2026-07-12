"""Reusable compact editor and review grid for note color families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from PySide6.QtCore import QSize, Signal, Qt
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

from synthesia2midi.core.color_families import morphology_for_slot, slots_for_family


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
        self._family_numbers: tuple[int, ...] = ()
        self._layout_mode: Literal["inline", "compact", "stacked"] | None = None
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setHorizontalSpacing(6)
        self._layout.setVerticalSpacing(3)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

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
                widget.hide()
                widget.deleteLater()

        self.rows.clear()
        self.remove_family_buttons.clear()
        self._family_headings.clear()
        self._family_numbers = family_numbers
        self._layout_mode = None
        if hasattr(self, "add_family_button"):
            del self.add_family_button

        for family_number in family_numbers:
            heading = QLabel(self.tr("Color {number}").format(number=family_number))
            heading.setWordWrap(True)
            heading.setStyleSheet("font-weight: bold;")
            self._family_headings[family_number] = heading

            if self.mode == "calibration" and family_number > 1:
                remove_button = self._remove_button(family_number)
                self.remove_family_buttons[family_number] = remove_button

            for slot in slots_for_family(family_number):
                self.rows[slot] = self._add_exemplar_row(
                    slot,
                    colors=colors,
                    enabled=enabled,
                    assignments=assignments,
                )

        if self.mode == "calibration" and len(family_numbers) < 4:
            self.add_family_button = QPushButton(self.tr("Add Color Family"))
            self.add_family_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.add_family_button.clicked.connect(self.family_add_requested)

        self._apply_layout_mode(self._layout_mode_for_width(self.width()))

    def _add_exemplar_row(
        self,
        slot: str,
        *,
        colors: Mapping[str, tuple[int, int, int] | None],
        enabled: Mapping[str, bool],
        assignments: Mapping[str, object],
    ) -> ExemplarRowWidgets:
        label_text = (
            self.tr("Natural")
            if morphology_for_slot(slot) == "natural"
            else self.tr("Sharp / Flat")
        )
        label = QLabel(label_text)
        label.setWordWrap(True)
        swatch = QLabel()
        swatch.setFixedSize(22, 20)

        assignment = assignments.get(slot)
        assignment_rgb = getattr(assignment, "rgb", None)
        rgb = assignment_rgb if assignment_rgb is not None else colors.get(slot)
        self._set_swatch_color(swatch, rgb)

        if self.mode == "review":
            status = QLabel(self.tr("Found") if assignment_rgb is not None else self.tr("Missing"))
            return ExemplarRowWidgets(label=label, swatch=swatch, status=status)

        set_button = QPushButton(self.tr("Set"))
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
        return ExemplarRowWidgets(
            label=label,
            swatch=swatch,
            set_button=set_button,
            present=present,
        )

    def _layout_mode_for_width(self, width: int) -> Literal["inline", "compact", "stacked"]:
        if not self.rows:
            return "inline"

        margins = self._layout.contentsMargins()
        frame_width = margins.left() + margins.right()
        spacing = self._layout.horizontalSpacing()
        label_width = max(row.label.sizeHint().width() for row in self.rows.values())
        swatch_width = max(row.swatch.sizeHint().width() for row in self.rows.values())

        if self.mode == "calibration":
            action_width = max(row.set_button.sizeHint().width() for row in self.rows.values())
            state_width = max(row.present.sizeHint().width() for row in self.rows.values())
        else:
            action_width = max(row.status.sizeHint().width() for row in self.rows.values())
            state_width = 0

        inline_width = (
            frame_width
            + label_width
            + swatch_width
            + action_width
            + state_width
            + spacing * (3 if self.mode == "calibration" else 2)
        )
        if width >= inline_width:
            return "inline"

        label_row_width = frame_width + label_width + swatch_width + spacing
        action_row_width = frame_width + action_width + state_width
        if self.mode == "calibration":
            action_row_width += spacing
        if hasattr(self, "add_family_button"):
            action_row_width = max(
                action_row_width,
                frame_width + self.add_family_button.sizeHint().width(),
            )
        if width >= max(label_row_width, action_row_width):
            return "compact"
        return "stacked"

    def _apply_layout_mode(self, mode: Literal["inline", "compact", "stacked"]) -> None:
        if mode == self._layout_mode:
            return

        while self._layout.count():
            self._layout.takeAt(0)
        for column in range(4):
            self._layout.setColumnMinimumWidth(column, 0)
            self._layout.setColumnStretch(column, 0)

        row_number = 0
        for family_number in self._family_numbers:
            heading = self._family_headings[family_number]
            remove_button = self.remove_family_buttons.get(family_number)
            column_count = 4 if mode == "inline" else 2
            heading_span = column_count - (1 if remove_button is not None else 0)
            self._layout.addWidget(heading, row_number, 0, 1, heading_span)
            if remove_button is not None:
                self._layout.addWidget(remove_button, row_number, column_count - 1)
            row_number += 1

            for slot in slots_for_family(family_number):
                row = self.rows[slot]
                if mode == "inline":
                    self._layout.addWidget(row.label, row_number, 0)
                    self._layout.addWidget(row.swatch, row_number, 1, Qt.AlignLeft | Qt.AlignVCenter)
                    if self.mode == "calibration":
                        self._layout.addWidget(row.set_button, row_number, 2)
                        self._layout.addWidget(row.present, row_number, 3)
                    else:
                        self._layout.addWidget(row.status, row_number, 2, 1, 2)
                    row_number += 1
                    continue

                self._layout.addWidget(row.label, row_number, 0)
                self._layout.addWidget(row.swatch, row_number, 1, Qt.AlignLeft | Qt.AlignVCenter)
                row_number += 1
                if self.mode == "review":
                    self._layout.addWidget(row.status, row_number, 0, 1, 2)
                    row_number += 1
                elif mode == "compact":
                    self._layout.addWidget(row.set_button, row_number, 0)
                    self._layout.addWidget(row.present, row_number, 1)
                    row_number += 1
                else:
                    self._layout.addWidget(row.set_button, row_number, 0, 1, 2)
                    row_number += 1
                    self._layout.addWidget(row.present, row_number, 0, 1, 2)
                    row_number += 1

        if hasattr(self, "add_family_button"):
            column_count = 4 if mode == "inline" else 2
            self._layout.addWidget(self.add_family_button, row_number, 0, 1, column_count)

        self._layout.setColumnStretch(0, 1)
        if mode != "inline":
            self._layout.setColumnStretch(1, 1)
        else:
            self._layout.setColumnStretch(3, 1)
        self._layout_mode = mode
        self._layout.invalidate()
        self._layout.activate()
        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_layout_mode(self._layout_mode_for_width(event.size().width()))

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
