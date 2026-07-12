"""Repo-root wrapper for the nested assisted-calibration probe."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_INNER_MODULE = _ROOT / "synthesia2midi" / "tools" / "probe_assisted_calibration.py"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "_synthesia2midi_probe_assisted_calibration",
    _INNER_MODULE,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load assisted-calibration probe from {_INNER_MODULE}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def __getattr__(name: str):
    return getattr(_MODULE, name)


def main() -> int:
    return _MODULE.main()


if __name__ == "__main__":
    raise SystemExit(main())
