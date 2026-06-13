"""Application version metadata for releases and packaged builds."""
from __future__ import annotations

import os
import re
from pathlib import Path

RELEASE_APP_NAME = "Synthesia2MIDI"
DEFAULT_APP_VERSION = "0.1.1-dev"
_VERSION_PATTERN = re.compile(r"^v?(?P<version>\d+\.\d+\.\d+(?:[-+._0-9A-Za-z]*)?)$")


def normalize_release_version(value: str | None) -> str | None:
    if not value:
        return None
    candidate = str(value).strip()
    if candidate.startswith("refs/tags/"):
        candidate = candidate.rsplit("/", 1)[-1]
    match = _VERSION_PATTERN.match(candidate)
    if not match:
        return None
    return match.group("version")


def _read_build_version_file() -> str | None:
    candidate = Path(__file__).with_name("build_version.txt")
    if not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def resolve_app_version(default: str = DEFAULT_APP_VERSION) -> str:
    for candidate in (
        os.getenv("S2M_RELEASE_VERSION"),
        _read_build_version_file(),
        os.getenv("GITHUB_REF_NAME"),
        os.getenv("GITHUB_REF"),
    ):
        normalized = normalize_release_version(candidate)
        if normalized:
            return normalized
    return default


APP_VERSION = resolve_app_version()
