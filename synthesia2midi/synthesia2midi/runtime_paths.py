"""Runtime path resolution for source checkouts and packaged builds."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _platform_binary_name(stem: str, *, platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return f"{stem}.exe"
    return stem


@dataclass(frozen=True)
class RuntimePaths:
    """Resolves helper binaries, assets, and writable locations."""

    frozen: bool
    app_root: Path
    repo_root: Path
    home_dir: Path
    platform_name: str = sys.platform

    @classmethod
    def detect(cls) -> "RuntimePaths":
        executable_root = Path(sys.executable).resolve().parent
        repo_root = Path(__file__).resolve().parents[2]
        return cls(
            frozen=bool(getattr(sys, "frozen", False)),
            app_root=executable_root if getattr(sys, "frozen", False) else repo_root,
            repo_root=repo_root,
            home_dir=Path.home(),
            platform_name=sys.platform,
        )

    @property
    def package_root(self) -> Path:
        return self.repo_root / "synthesia2midi"

    def ffmpeg_path(self) -> Path | None:
        return self._find_binary(
            "ffmpeg",
            self.app_root / "bin" / self.binary_name("ffmpeg"),
            self.repo_root / "ffmpeg" / self.binary_name("ffmpeg"),
        )

    def ffprobe_path(self) -> Path | None:
        return self._find_binary(
            "ffprobe",
            self.app_root / "bin" / self.binary_name("ffprobe"),
            self.repo_root / "ffmpeg" / self.binary_name("ffprobe"),
        )

    def deno_path(self) -> Path | None:
        return self._find_binary(
            "deno",
            self.app_root / "bin" / self.binary_name("deno"),
        )

    def rust_editor_path(self) -> Path | None:
        return self._first_executable(
            self.app_root / "bin" / self.binary_name("midi-touchup-editor"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / self.binary_name("midi-touchup-editor"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / self.binary_name("midi_touchup_editor_rust"),
        )

    def rust_soundfont_path(self) -> Path | None:
        return self._first_file(
            self.app_root / "assets" / "soundfonts" / "TouchUpPiano.sf2",
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano.sf2",
        )

    def rust_soundfont_license_path(self) -> Path | None:
        return self._first_file(
            self.app_root / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt",
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt",
        )

    def default_video_dir(self) -> Path:
        movies_dir = self.home_dir / "Movies"
        documents_dir = self.home_dir / "Documents"
        if movies_dir.exists():
            return movies_dir
        return documents_dir

    def log_dir(self) -> Path:
        if self.platform_name.startswith("win"):
            local_app_data = Path(os.getenv("LOCALAPPDATA") or (self.home_dir / "AppData" / "Local"))
            return local_app_data / "synthesia2midi" / "logs"
        if self.platform_name == "darwin":
            return self.home_dir / "Library" / "Logs" / "synthesia2midi"
        return self.home_dir / ".synthesia2midi" / "logs"

    def debug_dir(self) -> Path:
        return self.log_dir().parent / "debug"

    def binary_name(self, stem: str) -> str:
        return _platform_binary_name(stem, platform_name=self.platform_name)

    def _find_binary(self, command_name: str, *candidates: Path) -> Path | None:
        path = self._first_executable(*candidates)
        if path is not None:
            return path
        which_path = shutil.which(command_name)
        return Path(which_path) if which_path else None

    def _first_executable(self, *candidates: Path) -> Path | None:
        for candidate in candidates:
            if candidate.is_file() and (self.platform_name.startswith("win") or os.access(candidate, os.X_OK)):
                return candidate
        return None

    @staticmethod
    def _first_file(*candidates: Path) -> Path | None:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


def detect_runtime_paths() -> RuntimePaths:
    """Public helper for runtime path detection."""
    return RuntimePaths.detect()
