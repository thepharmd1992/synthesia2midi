"""Dialog placement helpers for keeping video overlays visible."""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget


def move_to_upper_left_safe_zone(dialog: QWidget, parent: QWidget | None = None) -> None:
    """Move a dialog near the upper-left screen area without covering title controls."""
    if not hasattr(dialog, "frameGeometry") or not hasattr(dialog, "move"):
        return

    screen = parent.screen() if parent is not None and hasattr(parent, "screen") else None
    if screen is None and hasattr(dialog, "screen"):
        screen = dialog.screen()
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is None:
        return

    available = screen.availableGeometry()
    frame = dialog.frameGeometry()
    width = max(1, frame.width())
    height = max(1, frame.height())

    x = min(
        max(available.left() + 24, available.left()),
        max(available.left(), available.right() - width),
    )
    y = min(
        max(available.top() + 48, available.top()),
        max(available.top(), available.bottom() - height),
    )
    dialog.move(x, y)
