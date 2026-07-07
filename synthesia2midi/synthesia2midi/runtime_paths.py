"""Runtime path resolution for source checkouts and packaged builds."""
from __future__ import annotations

import os
import hashlib
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


def _platform_binary_name(stem: str, *, platform_name: str | None = None) -> str:
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return f"{stem}.exe"
    return stem


def _safe_path_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug[:80] or "video"


def _path_hash(value: str) -> str:
    return hashlib.sha1(str(Path(value).expanduser()).encode("utf-8")).hexdigest()[:10]


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
            *self._bundle_binary_candidates("ffmpeg"),
            self.repo_root / "ffmpeg" / self.binary_name("ffmpeg"),
        )

    def ffprobe_path(self) -> Path | None:
        return self._find_binary(
            "ffprobe",
            *self._bundle_binary_candidates("ffprobe"),
            self.repo_root / "ffmpeg" / self.binary_name("ffprobe"),
        )

    def deno_path(self) -> Path | None:
        return self._find_binary(
            "deno",
            *self._bundle_binary_candidates("deno"),
        )

    def rust_editor_path(self) -> Path | None:
        return self._first_executable(
            *self._bundle_binary_candidates("midi-touchup-editor"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / self.binary_name("midi-touchup-editor"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / self.binary_name("midi_touchup_editor_rust"),
        )

    def rust_soundfont_path(self) -> Path | None:
        return self._first_file(
            *self._bundle_asset_candidates("soundfonts", "TouchUpPiano.sf2"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano.sf2",
        )

    def rust_soundfont_license_path(self) -> Path | None:
        return self._first_file(
            *self._bundle_asset_candidates("soundfonts", "TouchUpPiano_LICENSE.txt"),
            self.repo_root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt",
        )

    def default_video_dir(self) -> Path:
        movies_dir = self.home_dir / "Movies"
        documents_dir = self.home_dir / "Documents"
        if movies_dir.exists():
            return movies_dir
        return documents_dir

    def desktop_dir(self) -> Path:
        return self.home_dir / "Desktop"

    def downloads_dir(self) -> Path:
        return self.home_dir / "Downloads"

    def midi_exports_dir(self) -> Path:
        return self.desktop_dir() / "Synthesia2MIDI MIDI Files"

    def default_download_dir(self) -> Path:
        return self.downloads_dir() / "Synthesia2MIDI"

    def app_data_dir(self) -> Path:
        if self.platform_name.startswith("win"):
            base = Path(os.environ.get("LOCALAPPDATA", self.home_dir / "AppData" / "Local"))
            return base / "Synthesia2MIDI"
        if self.platform_name == "darwin":
            return self.home_dir / "Library" / "Application Support" / "Synthesia2MIDI"
        base = Path(os.environ.get("XDG_DATA_HOME", self.home_dir / ".local" / "share"))
        return base / "synthesia2midi"

    def project_data_dir(self) -> Path:
        return self.app_data_dir() / "projects"

    def project_slug_for_video(self, video_path: str) -> str:
        stem = Path(video_path).stem or "video"
        return f"{_safe_path_slug(stem)}-{_path_hash(video_path)}"

    def project_dir_for_video(self, video_path: str) -> Path:
        return self.project_data_dir() / self.project_slug_for_video(video_path)

    def project_ini_path(self, video_path: str) -> Path:
        return self.project_dir_for_video(video_path) / f"{Path(video_path).stem}.ini"

    def project_overlay_json_path(self, video_path: str) -> Path:
        return self.project_dir_for_video(video_path) / f"{Path(video_path).stem}_overlays.json"

    def project_frames_dir(self, video_path: str) -> Path:
        return self.project_dir_for_video(video_path) / f"{Path(video_path).stem}_frames"

    def conversion_settings_path(self, video_path: str, midi_path: str | Path) -> Path:
        return self.project_dir_for_video(video_path) / f"{Path(midi_path).stem}_settings.json"

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

    def _bundle_binary_candidates(self, stem: str) -> tuple[Path, ...]:
        binary_name = self.binary_name(stem)
        return tuple(root / "bin" / binary_name for root in self._bundle_roots())

    def _bundle_asset_candidates(self, *relative_parts: str) -> tuple[Path, ...]:
        return tuple(root / "assets" / Path(*relative_parts) for root in self._bundle_roots())

    def _bundle_roots(self) -> tuple[Path, ...]:
        candidates = [self.app_root]
        if self.frozen:
            candidates.append(self.repo_root)
            if self.platform_name == "darwin":
                contents_root = self.app_root.parent
                candidates.extend(
                    [
                        contents_root / "Frameworks",
                        contents_root / "Resources",
                    ]
                )

        deduped: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = Path(candidate)
            if resolved in seen:
                continue
            deduped.append(resolved)
            seen.add(resolved)
        return tuple(deduped)

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
