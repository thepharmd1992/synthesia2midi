# Packaged Release And YouTube Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build portable Windows x64 and macOS Apple Silicon releases that bundle the app plus required helpers, and harden the YouTube downloader with bundled JS runtime support and browser-cookie retry.

**Architecture:** Introduce one runtime-path layer that separates source-checkout behavior from packaged-app behavior, then route bundled helpers and user-writable directories through it. Harden the YouTube downloader on top of that runtime layer, then add packaging and release automation around the stabilized runtime contract.

**Tech Stack:** Python 3.12, PySide6, yt-dlp, PyInstaller, FFmpeg/ffprobe, Deno, Rust touch-up editor, GitHub Actions

---

## File Structure

- Create: `synthesia2midi/synthesia2midi/runtime_paths.py`
- Create: `synthesia2midi/synthesia2midi/version.py`
- Create: `tests/test_runtime_paths.py`
- Create: `tests/test_youtube_runtime_policy.py`
- Create: `tests/test_packaged_entrypoint.py`
- Create: `packaging/build_release.py`
- Create: `packaging/Synthesia2MIDI.spec`
- Create: `.github/workflows/release.yml`
- Modify: `synthesia2midi/run.py`
- Modify: `synthesia2midi/synthesia2midi/core/logging_config.py`
- Modify: `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`
- Modify: `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py`
- Modify: `synthesia2midi/synthesia2midi/youtube_downloader.py`
- Modify: `synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py`
- Modify: `synthesia2midi/synthesia2midi/utils/ffmpeg_helper.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/calibration.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`
- Modify: `README.md`
- Modify: `docs/testing.md`

### Task 1: Add Version And Runtime-Path Foundation

**Files:**
- Create: `synthesia2midi/synthesia2midi/version.py`
- Create: `synthesia2midi/synthesia2midi/runtime_paths.py`
- Test: `tests/test_runtime_paths.py`

- [ ] **Step 1: Write failing runtime-path tests**

```python
from pathlib import Path

from synthesia2midi.runtime_paths import RuntimePaths


def test_bundle_bin_prefers_frozen_bundle(tmp_path):
    bundle_root = tmp_path / "bundle"
    ffmpeg = bundle_root / "bin" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_text("", encoding="utf-8")
    ffmpeg.chmod(0o755)

    paths = RuntimePaths(
        frozen=True,
        bundle_root=bundle_root,
        source_root=tmp_path / "src",
        home_dir=tmp_path / "home",
    )

    assert paths.ffmpeg_path() == ffmpeg


def test_default_video_dir_uses_user_writable_location(tmp_path):
    paths = RuntimePaths(
        frozen=True,
        bundle_root=tmp_path / "bundle",
        source_root=tmp_path / "src",
        home_dir=tmp_path / "home",
    )

    assert "videos" not in str(paths.default_video_dir())
    assert paths.default_video_dir().is_absolute()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_runtime_paths.py -v`
Expected: FAIL with import or missing attribute errors for `RuntimePaths`

- [ ] **Step 3: Write minimal version source**

```python
APP_NAME = "Synthesia2MIDI"
APP_VERSION = "0.1.0-dev"
```

- [ ] **Step 4: Implement runtime-path resolver**

```python
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    frozen: bool
    bundle_root: Path
    source_root: Path
    home_dir: Path

    @classmethod
    def detect(cls) -> "RuntimePaths":
        frozen = bool(getattr(sys, "frozen", False))
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        source_root = Path(__file__).resolve().parents[2]
        home_dir = Path.home()
        return cls(frozen=frozen, bundle_root=bundle_root, source_root=source_root, home_dir=home_dir)

    def ffmpeg_path(self) -> Path | None:
        return self._first_existing(
            self.bundle_root / "bin" / self._platform_name("ffmpeg"),
            self.source_root / "ffmpeg" / self._platform_name("ffmpeg"),
        )

    def ffprobe_path(self) -> Path | None:
        return self._first_existing(
            self.bundle_root / "bin" / self._platform_name("ffprobe"),
            self.source_root / "ffmpeg" / self._platform_name("ffprobe"),
        )

    def deno_path(self) -> Path | None:
        return self._first_existing(self.bundle_root / "bin" / self._platform_name("deno"))

    def rust_editor_path(self) -> Path | None:
        return self._first_existing(
            self.bundle_root / "bin" / self._platform_name("midi-touchup-editor"),
            self.source_root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / self._platform_name("midi-touchup-editor"),
        )

    def rust_soundfont_path(self) -> Path | None:
        return self._first_existing(
            self.bundle_root / "assets" / "soundfonts" / "TouchUpPiano.sf2",
            self.source_root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano.sf2",
        )

    def default_video_dir(self) -> Path:
        base = self.home_dir / "Movies"
        return base if base.exists() else self.home_dir / "Documents"

    def log_dir(self) -> Path:
        if sys.platform == "darwin":
            return self.home_dir / "Library" / "Logs" / "synthesia2midi"
        if sys.platform == "win32":
            return Path(os.getenv("LOCALAPPDATA", self.home_dir / "AppData" / "Local")) / "synthesia2midi" / "logs"
        return self.home_dir / ".synthesia2midi" / "logs"

    def debug_dir(self) -> Path:
        return self.log_dir().parent / "debug"

    @staticmethod
    def _first_existing(*candidates: Path) -> Path | None:
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _platform_name(stem: str) -> str:
        return f"{stem}.exe" if sys.platform == "win32" else stem
```

- [ ] **Step 5: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_runtime_paths.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add synthesia2midi/synthesia2midi/version.py synthesia2midi/synthesia2midi/runtime_paths.py tests/test_runtime_paths.py
git commit -m "feat: add packaged runtime path resolver"
```

### Task 2: Route App Entry And Writable Paths Through Runtime Layer

**Files:**
- Modify: `synthesia2midi/run.py`
- Modify: `synthesia2midi/synthesia2midi/core/logging_config.py`
- Modify: `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`
- Modify: `synthesia2midi/synthesia2midi/workflows/calibration.py`
- Test: `tests/test_packaged_entrypoint.py`

- [ ] **Step 1: Write failing packaged-entrypoint tests**

```python
from synthesia2midi.runtime_paths import RuntimePaths


def test_logging_uses_runtime_log_dir(monkeypatch, tmp_path):
    from synthesia2midi.core.logging_config import _default_log_dir

    fake_paths = RuntimePaths(
        frozen=True,
        bundle_root=tmp_path / "bundle",
        source_root=tmp_path / "src",
        home_dir=tmp_path / "home",
    )
    monkeypatch.setattr("synthesia2midi.core.logging_config.detect_runtime_paths", lambda: fake_paths)

    assert _default_log_dir() == str(fake_paths.log_dir())
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_packaged_entrypoint.py -v`
Expected: FAIL because `detect_runtime_paths` or runtime-based path logic does not exist

- [ ] **Step 3: Update packaged entrypoint to use runtime layer**

```python
from synthesia2midi.runtime_paths import detect_runtime_paths
from synthesia2midi.version import APP_NAME, APP_VERSION

runtime_paths = detect_runtime_paths()
log_file = LoggingConfig.setup_logging(
    log_to_file=True,
    log_to_console=not runtime_paths.frozen,
    log_level=logging.INFO,
    log_dir=str(runtime_paths.log_dir()),
)
```

- [ ] **Step 4: Replace repo-root defaults with runtime defaults**

```python
runtime_paths = detect_runtime_paths()
videos_dir = runtime_paths.default_video_dir()
dialog.setDirectory(str(videos_dir))
screenshot_dir = runtime_paths.debug_dir()
```

- [ ] **Step 5: Remove packaged dependence on `my_immortal.ini`**

```python
template_ini_path = runtime_paths.source_root / "synthesia2midi" / "my_immortal.ini"
if template_ini_path.exists():
    logging.info("Applying overlay template styles from %s", template_ini_path)
    config_manager = ConfigManager(self.app_state)
    template_overlays = config_manager.parse_overlays_from_file(str(template_ini_path))
    self.apply_template_style(self.app_state.overlays, template_overlays)
else:
    self.apply_template_style(self.app_state.overlays, None)
```

- [ ] **Step 6: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_packaged_entrypoint.py tests/test_recent_videos.py tests/test_video2midi_app_smoke.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add synthesia2midi/run.py synthesia2midi/synthesia2midi/core/logging_config.py synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py synthesia2midi/synthesia2midi/main.py synthesia2midi/synthesia2midi/workflows/calibration.py tests/test_packaged_entrypoint.py
git commit -m "feat: route packaged app paths through runtime layer"
```

### Task 3: Bundle-Aware FFmpeg, ffprobe, Rust Editor, And Soundfont Routing

**Files:**
- Modify: `synthesia2midi/synthesia2midi/utils/ffmpeg_helper.py`
- Modify: `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`
- Test: `tests/test_setup_and_launch.py`
- Test: `tests/test_midi_touchup_controller.py`

- [ ] **Step 1: Add failing helper-resolution tests**

```python
def test_find_ffmpeg_prefers_runtime_bundle(monkeypatch, tmp_path):
    from synthesia2midi import runtime_paths
    from synthesia2midi.utils import ffmpeg_helper

    ffmpeg = tmp_path / "bundle" / "bin" / "ffmpeg"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_text("", encoding="utf-8")
    ffmpeg.chmod(0o755)
    monkeypatch.setattr(runtime_paths, "detect_runtime_paths", lambda: runtime_paths.RuntimePaths(True, ffmpeg.parents[1], tmp_path / "src", tmp_path / "home"))

    assert ffmpeg_helper.find_ffmpeg() == str(ffmpeg)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_setup_and_launch.py tests/test_midi_touchup_controller.py -v`
Expected: FAIL because helpers still rely on PATH or repo-root discovery

- [ ] **Step 3: Route FFmpeg and ffprobe through runtime paths**

```python
from synthesia2midi.runtime_paths import detect_runtime_paths


def find_ffmpeg() -> Optional[str]:
    runtime_path = detect_runtime_paths().ffmpeg_path()
    if runtime_path is not None:
        return str(runtime_path)
    return shutil.which("ffmpeg")


def find_ffprobe() -> Optional[str]:
    runtime_path = detect_runtime_paths().ffprobe_path()
    if runtime_path is not None:
        return str(runtime_path)
    return shutil.which("ffprobe")
```

- [ ] **Step 4: Route Rust editor through runtime paths and suppress repo-only setup messaging in packaged mode**

```python
runtime_paths = detect_runtime_paths()
binary_path = runtime_paths.rust_editor_path()
if binary_path is None and runtime_paths.frozen:
    QMessageBox.warning(
        self.app,
        "Touch-Up Editor Missing",
        "Bundled touch-up editor files were not found in this app build.",
    )
```

- [ ] **Step 5: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_setup_and_launch.py tests/test_midi_touchup_controller.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add synthesia2midi/synthesia2midi/utils/ffmpeg_helper.py synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py tests/test_setup_and_launch.py tests/test_midi_touchup_controller.py
git commit -m "feat: resolve bundled helper binaries at runtime"
```

### Task 4: Harden YouTube Runtime Wiring And Browser-Cookie Retry

**Files:**
- Modify: `synthesia2midi/synthesia2midi/youtube_downloader.py`
- Modify: `synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py`
- Create: `tests/test_youtube_runtime_policy.py`
- Modify: `tests/test_youtube_downloader.py`

- [ ] **Step 1: Write failing YouTube policy tests**

```python
def test_retry_policy_uses_preferred_browser_for_auth_failures():
    from synthesia2midi.youtube_downloader import should_retry_with_browser_cookies

    assert should_retry_with_browser_cookies("Sign in to confirm your age") is True


def test_browser_cookie_args_use_preferred_browser():
    from synthesia2midi.youtube_downloader import browser_cookie_args

    assert browser_cookie_args("chrome") == ("chrome",)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv/bin/python -m pytest tests/test_youtube_runtime_policy.py tests/test_youtube_downloader.py -v`
Expected: FAIL because retry-policy helpers and browser-cookie path do not exist

- [ ] **Step 3: Add explicit runtime configuration for yt-dlp**

```python
def _youtube_ydl_opts(base_opts: Dict[str, Any]) -> Dict[str, Any]:
    opts = dict(base_opts)
    runtime_paths = detect_runtime_paths()
    deno_path = runtime_paths.deno_path()
    if deno_path is not None:
        opts["js_runtimes"] = {"deno": {"path": str(deno_path)}}
        opts["remote_components"] = ["ejs:github"]
    ffmpeg_path = runtime_paths.ffmpeg_path()
    if ffmpeg_path is not None:
        opts["ffmpeg_location"] = str(ffmpeg_path.parent)
    return opts
```

- [ ] **Step 4: Add preferred-browser persistence and auto-retry**

```python
SUPPORTED_COOKIE_BROWSERS = ("chrome", "edge", "safari")


def should_retry_with_browser_cookies(message: str) -> bool:
    normalized = message.lower()
    return any(
        token in normalized
        for token in ("sign in", "cookies", "age", "bot", "challenge", "javascript runtime")
    )
```

```python
try:
    return self._download_once(url, quality=quality, progress_hook=progress_hook, overwrite=overwrite)
except Exception as exc:
    if self.auto_cookie_retry and should_retry_with_browser_cookies(str(exc)):
        return self._download_once(
            url,
            quality=quality,
            progress_hook=progress_hook,
            overwrite=overwrite,
            browser_cookie=self.preferred_browser,
        )
    raise
```

- [ ] **Step 5: Surface retry status in the dialog**

```python
self.progress_handler.status.emit(f"Retrying with {preferred_browser.title()} browser cookies...")
```

- [ ] **Step 6: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_youtube_runtime_policy.py tests/test_youtube_downloader.py tests/test_youtube_download_dialog.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add synthesia2midi/synthesia2midi/youtube_downloader.py synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py tests/test_youtube_runtime_policy.py tests/test_youtube_downloader.py tests/test_youtube_download_dialog.py
git commit -m "feat: harden YouTube downloader retries in packaged builds"
```

### Task 5: Add PyInstaller Packaging And Local Build Script

**Files:**
- Create: `packaging/build_release.py`
- Create: `packaging/Synthesia2MIDI.spec`
- Modify: `synthesia2midi/synthesia2midi/version.py`
- Test: ad hoc local packaged build

- [ ] **Step 1: Add packaging smoke test command to plan and verify PyInstaller is absent**

Run: `.venv/bin/python -m PyInstaller --version`
Expected: FAIL before adding packaging dependency

- [ ] **Step 2: Create build script**

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    run(["cargo", "build", "--release"], cwd=ROOT / "tools" / "midi_touchup_editor_rust")
    run([str(ROOT / ".venv" / "bin" / "python"), "-m", "pip", "install", "pyinstaller"])
    run([str(ROOT / ".venv" / "bin" / "python"), "-m", "PyInstaller", str(ROOT / "packaging" / "Synthesia2MIDI.spec"), "--noconfirm"])
    shutil.make_archive(str(DIST / "Synthesia2MIDI-local"), "zip", DIST, "Synthesia2MIDI")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create PyInstaller spec**

```python
from pathlib import Path

root = Path(SPECPATH).resolve().parents[1]
datas = [
    (str(root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano.sf2"), "assets/soundfonts"),
    (str(root / "tools" / "midi_touchup_editor_rust" / "assets" / "soundfonts" / "TouchUpPiano_LICENSE.txt"), "assets/soundfonts"),
]
binaries = [
    (str(root / "tools" / "midi_touchup_editor_rust" / "target" / "release" / ("midi-touchup-editor.exe" if is_win else "midi-touchup-editor")), "bin"),
]
```

- [ ] **Step 4: Run local packaging build**

Run: `.venv/bin/python packaging/build_release.py`
Expected: bundled `dist/Synthesia2MIDI` output plus zipped archive

- [ ] **Step 5: Smoke-launch packaged app**

Run: `QT_QPA_PLATFORM=offscreen dist/Synthesia2MIDI/Synthesia2MIDI`
Expected: app starts without repo-root setup errors

- [ ] **Step 6: Commit**

```bash
git add packaging/build_release.py packaging/Synthesia2MIDI.spec synthesia2midi/synthesia2midi/version.py
git commit -m "feat: add local packaged release build pipeline"
```

### Task 6: Add Tag-Based Release Workflow And Docs

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `README.md`
- Modify: `docs/testing.md`

- [ ] **Step 1: Write workflow with tag-only trigger**

```yaml
name: Release

on:
  push:
    tags:
      - "v*"
```

- [ ] **Step 2: Add matrix builds and artifact publication**

```yaml
jobs:
  build-release:
    strategy:
      matrix:
        include:
          - os: macos-latest
            artifact_name: Synthesia2MIDI-macos-arm64
          - os: windows-latest
            artifact_name: Synthesia2MIDI-windows-x64
```

- [ ] **Step 3: Update user-facing docs**

```markdown
## Download

End users should download packaged builds from GitHub Releases.

- macOS: unzip, open the app, and use `Open Anyway` if Gatekeeper blocks launch
- Windows: unzip, launch the app, and use `Run anyway` if SmartScreen warns
```

- [ ] **Step 4: Run lint and focused workflow sanity check**

Run: `.venv/bin/python -m ruff check synthesia2midi tests packaging`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml README.md docs/testing.md
git commit -m "ci: add packaged release workflow and user docs"
```

### Task 7: Full Verification Gate

**Files:**
- Verify all modified files

- [ ] **Step 1: Run default gate**

Run: `git diff --check && .venv/bin/python -m compileall -q synthesia2midi && .venv/bin/python -m pytest`
Expected: PASS

- [ ] **Step 2: Run packaged-build verification**

Run: `.venv/bin/python packaging/build_release.py`
Expected: PASS with generated packaged artifact

- [ ] **Step 3: Smoke test packaged helper routing**

Run: `QT_QPA_PLATFORM=offscreen dist/Synthesia2MIDI/Synthesia2MIDI`
Expected: launch succeeds without `.venv`, `setup_env.py`, or repo-root errors

- [ ] **Step 4: Check git status is clean except expected packaged artifacts ignored by git**

Run: `git status --short`
Expected: no unexpected tracked-file drift

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: ship packaged release and hardened YouTube runtime"
```
