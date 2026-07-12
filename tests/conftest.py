import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "synthesia2midi"
package_root = str(PACKAGE_ROOT)
if package_root in sys.path:
    sys.path.remove(package_root)
sys.path.insert(0, package_root)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
