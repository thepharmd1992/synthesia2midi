import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "synthesia2midi"
package_root = str(PACKAGE_ROOT)
if package_root in sys.path:
    sys.path.remove(package_root)
sys.path.insert(0, package_root)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def flush_qt_deferred_deletes():
    """Keep PySide wrappers from accumulating until pytest's final GC pass."""
    yield
    app = QApplication.instance()
    if app is None:
        return
    app.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
