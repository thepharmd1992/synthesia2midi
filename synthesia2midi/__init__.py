"""Repository-root import shim for the nested Synthesia2MIDI package layout."""
from __future__ import annotations

from pathlib import Path

_NESTED_PACKAGE = Path(__file__).resolve().parent / "synthesia2midi"
if _NESTED_PACKAGE.is_dir():
    __path__.append(str(_NESTED_PACKAGE))
