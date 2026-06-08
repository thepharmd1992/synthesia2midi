"""Recent video file persistence for app-level startup history."""
from __future__ import annotations

import os
from collections.abc import Iterable

from PySide6.QtCore import QSettings


RECENT_VIDEO_FILES_KEY = "recent_video_files"
DEFAULT_MAX_RECENT_VIDEOS = 5


class RecentVideoStore:
    """Stores recently opened file-picker paths outside per-video config."""

    def __init__(self, *, settings=None, max_entries: int = DEFAULT_MAX_RECENT_VIDEOS) -> None:
        self._settings = settings or QSettings("Synthesia2MIDI", "Synthesia2MIDI")
        self._max_entries = int(max_entries)

    def recent_paths(self) -> list[str]:
        paths = self._dedupe_existing_paths(self._stored_paths())
        paths = paths[: self._max_entries]
        self._settings.setValue(RECENT_VIDEO_FILES_KEY, paths)
        return paths

    def add(self, path: str) -> None:
        normalized = self._normalize_path(path)
        if not normalized or not os.path.exists(normalized):
            return

        paths = [candidate for candidate in self.recent_paths() if candidate != normalized]
        paths.insert(0, normalized)
        self._settings.setValue(RECENT_VIDEO_FILES_KEY, paths[: self._max_entries])

    def _stored_paths(self) -> list[str]:
        value = self._settings.value(RECENT_VIDEO_FILES_KEY, [])
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Iterable):
            return [str(path) for path in value]
        return []

    def _dedupe_existing_paths(self, paths: Iterable[str]) -> list[str]:
        deduped = []
        seen = set()
        for path in paths:
            normalized = self._normalize_path(path)
            if not normalized or normalized in seen or not os.path.exists(normalized):
                continue
            deduped.append(normalized)
            seen.add(normalized)
        return deduped

    @staticmethod
    def _normalize_path(path: str) -> str:
        if not path:
            return ""
        return os.path.abspath(os.path.expanduser(str(path)))
