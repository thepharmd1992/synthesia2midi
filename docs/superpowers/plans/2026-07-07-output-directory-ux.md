# Output Directory UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated files land in places that make sense to normal users: final MIDI files in a dedicated Desktop folder, downloaded source videos in Downloads, and technical working files in an app-managed project folder. Keep existing user installs compatible by reading old sidecar config, overlay, and frame locations as fallbacks.

**Architecture:** Add platform-aware output locations to the existing runtime path layer, then route MIDI export, YouTube download defaults, frame extraction, config save/load, and conversion-complete UI through those paths. New writes use the new layout. Old files remain readable through explicit fallback search paths. GUI changes stay in Qt controllers; workflow code receives paths from runtime helpers instead of inventing filesystem locations locally.

**Tech Stack:** Python 3, PySide6, pytest, Qt translation files, existing Synthesia2MIDI runtime path and workflow/controller modules.

## Global Constraints

- Do not push.
- Do not create a worktree.
- Start implementation with `git status --short --branch` and preserve unrelated changes.
- Keep backwards compatibility for existing installed users:
  - New writes use the new layout.
  - Old sidecar `.ini`, `_overlays.json`, and `_frames/` paths remain readable.
  - Do not silently move old files.
- Do not put generated frame images, logs, downloaded videos, or MIDI outputs under version control.
- Do not make broad refactors in `main.py`; keep changes in focused helpers, workflows, and controllers.
- Update localization/audit artifacts for new user-visible text.
- Use the existing `RuntimePaths` style rather than adding a second unrelated path system.

---

## Desired User-Facing Behavior

- Final MIDI files save by default to:
  - macOS/Linux/Windows user Desktop folder, inside `Synthesia2MIDI MIDI Files`
- Downloaded YouTube videos save by default to:
  - user Downloads folder, inside `Synthesia2MIDI/(per-video slug)/`
- Working project files save to app data:
  - macOS: `~/Library/Application Support/Synthesia2MIDI/projects/(project slug)/`
  - Windows: `%LOCALAPPDATA%/Synthesia2MIDI/projects/(project slug)/`
  - Linux: `$XDG_DATA_HOME/synthesia2midi/projects/(project slug)/` or `~/.local/share/synthesia2midi/projects/(project slug)/`
- Working project files include:
  - extracted frames
  - video `.ini`
  - `_overlays.json`
  - conversion settings JSON
- Conversion completion dialog has two explicit choices:
  - `Open Touch-Up Editor`
  - `Show MIDI in Folder`
- `Done` is removed from the dialog buttons. Closing the dialog with the window close control can still dismiss it.

## Files Expected To Change

```text
backlog/tasks/(new Improve output directory UX task)
synthesia2midi/synthesia2midi/runtime_paths.py
synthesia2midi/synthesia2midi/config_manager.py
synthesia2midi/synthesia2midi/workflows/video_loading.py
synthesia2midi/synthesia2midi/workflows/video_to_frames.py
synthesia2midi/synthesia2midi/workflows/conversion.py
synthesia2midi/synthesia2midi/workflows/midi_export.py
synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py
synthesia2midi/synthesia2midi/gui/midi_conversion_controller.py
synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py
synthesia2midi/synthesia2midi/gui/youtube_download_dialog.py
synthesia2midi/synthesia2midi/translations/synthesia2midi_*.ts
synthesia2midi/synthesia2midi/translations/synthesia2midi_*.qm
docs/localization/ui-string-manifest.json
tests/test_runtime_paths.py
tests/test_config_manager.py
tests/test_midi_conversion_controller.py
tests/test_midi_touchup_controller.py
tests/test_video_loading_paths.py
tests/test_video_to_frames_controller.py
tests/test_youtube_download_dialog.py
tests/test_packaged_entrypoint.py
```

The exact Backlog task number comes from the next available task number in `backlog/tasks/`.

---

## Step 1: Create Backlog Task And Baseline

- [x] Run:

```bash
git status --short --branch
ls backlog/tasks | sort | tail -20
```

- [x] Create one Backlog task with acceptance criteria matching this plan.

Expected task content shape:

```markdown
# Improve output directory UX

## Status
Todo

## Context
Users currently see working files and final MIDI output mixed into media-oriented folders. The final MIDI should be easy to find, while frame/config/project data should be kept in app-managed storage.

## Acceptance Criteria
- Final MIDI files default to `Desktop/Synthesia2MIDI MIDI Files/`.
- YouTube downloads default to `Downloads/Synthesia2MIDI/(per-video slug)/`.
- Frame series, config, overlays, and conversion settings are written under app-managed project data.
- Old sidecar config, overlays, and frame folders still load.
- The conversion-complete dialog offers `Open Touch-Up Editor` and `Show MIDI in Folder`.
- Tests cover the new path helpers, legacy fallbacks, and dialog behavior.
- Localization/audit artifacts are updated for new visible strings.
```

- [x] Run the focused pre-change tests that currently describe the old behavior:

```bash
.venv/bin/python -m pytest tests/test_runtime_paths.py tests/test_midi_conversion_controller.py tests/test_packaged_entrypoint.py
```

Expected result before edits: tests pass on current behavior.

Commit after this step:

```bash
git add backlog/tasks docs/superpowers/plans/2026-07-07-output-directory-ux.md
git commit -m "Add output directory UX backlog task"
```

---

## Step 2: Add Runtime Path Helpers

Implement the path policy in one place.

- [x] Add methods to `synthesia2midi/synthesia2midi/runtime_paths.py`.

Core behavior to implement:

```python
import hashlib
import os
import re
from pathlib import Path


def _safe_path_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug[:80] or "video"


def _path_hash(value: str) -> str:
    return hashlib.sha1(str(Path(value).expanduser()).encode("utf-8")).hexdigest()[:10]
```

Add these `RuntimePaths` methods:

```python
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
```

- [x] Keep `default_video_dir()` for open-file picker compatibility, but stop using it for downloads and MIDI export.
- [x] Add tests in `tests/test_runtime_paths.py`:

```python
def test_user_output_dirs_are_dedicated_folders(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )

    assert paths.midi_exports_dir() == tmp_path / "Desktop" / "Synthesia2MIDI MIDI Files"
    assert paths.default_download_dir() == tmp_path / "Downloads" / "Synthesia2MIDI"


def test_project_paths_are_app_managed_and_stable(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    video_path = "/Users/jeff/Movies/Game of Thrones Main Theme.mp4"

    project_dir = paths.project_dir_for_video(video_path)

    assert project_dir.parent.name == "projects"
    assert "game-of-thrones-main-theme" in project_dir.name
    assert paths.project_ini_path(video_path).parent == project_dir
    assert paths.project_overlay_json_path(video_path).parent == project_dir
    assert paths.project_frames_dir(video_path).parent == project_dir
```

- [x] Add a platform-specific test for macOS app data using `platform_name`:

```python
def test_project_data_dir_uses_macos_application_support(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )

    assert paths.project_data_dir() == (
        tmp_path / "Library" / "Application Support" / "Synthesia2MIDI" / "projects"
    )
```

- [x] Run:

```bash
.venv/bin/python -m pytest tests/test_runtime_paths.py
```

Expected result: all runtime path tests pass.

Commit after this step:

```bash
git add synthesia2midi/synthesia2midi/runtime_paths.py tests/test_runtime_paths.py
git commit -m "Add app output path helpers"
```

---

## Step 3: Route MIDI Export To Desktop Folder

Make final MIDI files easy to find and stop putting conversion settings next to them.

- [x] Update `synthesia2midi/synthesia2midi/workflows/midi_export.py` so `MidiExportService` accepts an optional `RuntimePaths`.

Implementation shape:

```python
from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


class MidiExportService:
    def __init__(self, app_state, conversion_workflow, runtime_paths: RuntimePaths | None = None):
        self.app_state = app_state
        self.conversion_workflow = conversion_workflow
        self.runtime_paths = runtime_paths or detect_runtime_paths()

    def _build_default_output_path(self) -> str:
        video_path = self.app_state.video.original_video_path or self.app_state.video.filepath
        stem = Path(video_path).stem if video_path else "synthesia2midi"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.runtime_paths.midi_exports_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{stem}_{timestamp}.mid")
```

Preserve the existing timestamp format exactly: `YYYYMMDD_HHMMSS`.

- [x] Update `synthesia2midi/synthesia2midi/gui/midi_conversion_controller.py` to pass a runtime path dependency when the app exposes one:

```python
runtime_paths = getattr(app, "runtime_paths", None)
result = MidiExportService(app.app_state, app.conversion_workflow, runtime_paths=runtime_paths).export_to_default_path()
```

- [x] Update `synthesia2midi/synthesia2midi/workflows/conversion.py` so `ConversionWorkflow` can receive `runtime_paths` and `_save_midi_settings_log()` writes to the project folder:

```python
from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


def __init__(
    self,
    app_state: AppState,
    video_session: VideoSession,
    parent_widget=None,
    detection_manager=None,
    runtime_paths: RuntimePaths | None = None,
):
    self.app_state = app_state
    self.video_session = video_session
    self.parent_widget = parent_widget
    self.detection_manager = detection_manager
    self.runtime_paths = runtime_paths or detect_runtime_paths()
```

```python
video_path = self.app_state.video.original_video_path or self.app_state.video.filepath
if video_path:
    settings_path = self.runtime_paths.conversion_settings_path(video_path, midi_path)
else:
    settings_path = Path(midi_path).with_suffix("").with_name(f"{Path(midi_path).stem}_settings.json")
settings_path.parent.mkdir(parents=True, exist_ok=True)
```

The fallback for missing `video_path` keeps the workflow from failing in synthetic tests.

- [x] Update `tests/test_midi_conversion_controller.py` expected path:

```python
expected_output = (
    tmp_path
    / "Desktop"
    / "Synthesia2MIDI MIDI Files"
    / "song_20260512_101500.mid"
)
```

In the controller tests, give the fake app deterministic paths:

```python
runtime_paths = RuntimePaths(
    frozen=False,
    app_root=tmp_path / "repo",
    repo_root=tmp_path / "repo",
    home_dir=tmp_path,
    platform_name="darwin",
)
app.runtime_paths = runtime_paths
```

In the service test, pass the same object directly:

```python
result = MidiExportService(app_state, workflow, runtime_paths=runtime_paths).export_to_default_path()
```

- [x] Add or update a conversion workflow test proving the settings JSON writes under `project_dir_for_video()` instead of the MIDI output folder.
- [x] Run:

```bash
.venv/bin/python -m pytest tests/test_midi_conversion_controller.py
```

Expected result: MIDI export tests pass and expected output path is the dedicated Desktop folder.

Commit after this step:

```bash
git add synthesia2midi/synthesia2midi/workflows/midi_export.py synthesia2midi/synthesia2midi/workflows/conversion.py synthesia2midi/synthesia2midi/gui/midi_conversion_controller.py tests/test_midi_conversion_controller.py
git commit -m "Save MIDI exports to Desktop output folder"
```

---

## Step 4: Replace Completion Done Button With Show-In-Folder

Give users an obvious next action after conversion.

- [x] Update `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`.

Implementation shape:

```python
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox
```

Add:

```python
def _show_midi_in_folder(self, midi_path: str) -> None:
    path = Path(midi_path)
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", str(path)], check=False)
        return
    if sys.platform.startswith("win"):
        subprocess.run(["explorer", f"/select,{os.path.normpath(path)}"], check=False)
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
```

Change `show_conversion_complete_dialog()` so it creates only:

```python
open_editor_button = msg_box.addButton(self.tr("Open Touch-Up Editor"), QMessageBox.AcceptRole)
show_folder_button = msg_box.addButton(self.tr("Show MIDI in Folder"), QMessageBox.ActionRole)
```

Then handle:

```python
clicked = msg_box.clickedButton()
if clicked == open_editor_button:
    self.open_editor_for_midi(midi_path)
elif clicked == show_folder_button:
    self._show_midi_in_folder(midi_path)
```

- [x] Add tests in `tests/test_midi_touchup_controller.py`:

```python
def test_conversion_complete_dialog_has_editor_and_show_folder(qtbot, monkeypatch, tmp_path):
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")
    controller = MidiTouchupController(_fake_app())
    shown_actions = []

    class FakeMessageBox:
        AcceptRole = QMessageBox.AcceptRole
        ActionRole = QMessageBox.ActionRole

        def __init__(self, parent=None):
            self.buttons = []
            self._clicked = None

        def setIcon(self, icon):
            self.icon = icon

        def setWindowTitle(self, title):
            self.window_title = title

        def setText(self, text):
            self.text = text

        def setInformativeText(self, text):
            self.informative_text = text

        def setDefaultButton(self, button):
            self.default_button = button

        def addButton(self, text, role):
            self.buttons.append((text, role))
            button = object()
            if text == "Show MIDI in Folder":
                self._clicked = button
            return button

        def clickedButton(self):
            return self._clicked

        def exec(self):
            shown_actions.extend(text for text, _role in self.buttons)

    monkeypatch.setattr("synthesia2midi.gui.midi_touchup_controller.QMessageBox", FakeMessageBox)
    monkeypatch.setattr(controller, "_show_midi_in_folder", lambda path: shown_actions.append("revealed"))

    controller.show_conversion_complete_dialog(str(midi_path))

    assert "Open Touch-Up Editor" in shown_actions
    assert "Show MIDI in Folder" in shown_actions
    assert "Done" not in shown_actions
    assert "revealed" in shown_actions
```

- [x] Add a platform-targeted test for macOS reveal command:

```python
def test_show_midi_in_folder_uses_macos_reveal(monkeypatch, tmp_path):
    midi_path = tmp_path / "song.mid"
    calls = []
    controller = MidiTouchupController(_fake_app())

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(subprocess, "run", lambda args, check=False: calls.append((args, check)))

    controller._show_midi_in_folder(str(midi_path))

    assert calls == [(["open", "-R", str(midi_path)], False)]
```

- [x] Run:

```bash
.venv/bin/python -m pytest tests/test_midi_touchup_controller.py tests/test_ui_string_audit.py
```

Expected result: touch-up controller tests pass. The UI string audit may fail until Step 8 if the manifest has not been updated; that failure should only list the new intentional string.

Observed at this checkpoint: `tests/test_midi_touchup_controller.py` passed, and `tests/test_ui_string_audit.py` failed only because the tracked manifest is stale after adding new visible dialog text.

Commit after this step:

```bash
git add synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py tests/test_midi_touchup_controller.py
git commit -m "Show MIDI location after conversion"
```

---

## Step 5: Move Config, Overlay, And Frame Writes To Project Storage

Keep technical files out of user media folders while still loading old installed-user files.

- [ ] Update `synthesia2midi/synthesia2midi/config_manager.py`.

Add `runtime_paths` injection without breaking existing callers:

```python
from pathlib import Path

from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


class ConfigManager:
    def __init__(self, app_state: AppState, runtime_paths: RuntimePaths | None = None):
        self.app_state = app_state
        self.runtime_paths = runtime_paths or detect_runtime_paths()
```

Use project paths for new saves:

```python
def _get_ini_path(self, video_filepath: str) -> str:
    return str(self.runtime_paths.project_ini_path(video_filepath)) if video_filepath else ""


def _get_overlay_json_path(self, video_filepath: str) -> str:
    return str(self.runtime_paths.project_overlay_json_path(video_filepath)) if video_filepath else ""
```

Add legacy helpers:

```python
def legacy_ini_paths_for_video(self, video_filepath: str) -> list[Path]:
    video_path = Path(video_filepath)
    base = video_path.with_suffix("")
    return [
        video_path.with_suffix(".ini"),
        Path(f"{base}_config.ini"),
        Path(f"{base}.config"),
        video_path.parent / "config.ini",
    ]


def legacy_overlay_json_path_for_video(self, video_filepath: str) -> Path:
    video_path = Path(video_filepath)
    return video_path.with_name(f"{video_path.stem}_overlays.json")


def config_candidates_for_video(self, video_filepath: str) -> list[Path]:
    return [self.runtime_paths.project_ini_path(video_filepath), *self.legacy_ini_paths_for_video(video_filepath)]
```

Ensure save creates parent directories:

```python
ini_path = self._get_ini_path(video_filepath)
Path(ini_path).parent.mkdir(parents=True, exist_ok=True)
```

Ensure overlay save creates parent directories:

```python
overlay_path = self._get_overlay_json_path(video_filepath)
Path(overlay_path).parent.mkdir(parents=True, exist_ok=True)
```

- [ ] Split overlay loading so project and legacy JSON files can be loaded directly from the INI location:

```python
def _load_overlay_data_from_path(self, overlay_json_path: Path) -> bool:
    if not overlay_json_path.exists():
        logging.info("Overlay JSON file not found: %s", overlay_json_path)
        return False
    with overlay_json_path.open("r", encoding="utf-8") as f:
        overlay_data = json.load(f)
    return self._apply_overlay_data(overlay_data, str(overlay_json_path))
```

In `load_config()`, try `Path(config_filepath).with_name(f"{Path(config_filepath).stem}_overlays.json")` before deriving paths from the stored video. This keeps old sidecar and new project `.ini` files paired with the overlay JSON next to them.

- [ ] Update `synthesia2midi/synthesia2midi/workflows/video_loading.py` so config discovery checks new project config first, then legacy config paths:

```python
from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


def __init__(
    self,
    app_state: AppState,
    config_manager: ConfigManager,
    parent_widget=None,
    runtime_paths: RuntimePaths | None = None,
):
    self.app_state = app_state
    self.config_manager = config_manager
    self.parent_widget = parent_widget
    self.runtime_paths = runtime_paths or getattr(config_manager, "runtime_paths", detect_runtime_paths())


for config_path in self.config_manager.config_candidates_for_video(video_filepath):
    if config_path.exists() and self.config_manager.load_config(str(config_path), is_template=False):
        self.app_state.video.filepath_ini_used = str(config_path)
        return True
```

- [ ] Update `save_current_config()` so `filepath_ini_used` records the new project `.ini`, not the old sidecar path:

```python
video_path_for_config = self.app_state.video.original_video_path or self.app_state.video.filepath
output_path = str(self.config_manager.runtime_paths.project_ini_path(video_path_for_config))
success = self.config_manager.save_config(video_path_for_config)
if success:
    self.app_state.video.filepath_ini_used = output_path
```

- [ ] Update frame extraction in `video_loading.py` and `video_to_frames.py`:

```python
from synthesia2midi.runtime_paths import RuntimePaths, detect_runtime_paths


def _legacy_frames_dir(video_path: str) -> Path:
    path = Path(video_path)
    return path.with_name(f"{path.stem}_frames")


def _frames_dir_for_video(self, video_path: str) -> Path:
    project_frames = self.runtime_paths.project_frames_dir(video_path)
    legacy_frames = _legacy_frames_dir(video_path)
    if project_frames.exists():
        return project_frames
    if legacy_frames.exists():
        return legacy_frames
    return project_frames
```

When creating frames, use the returned directory and create parents.

For `VideoToFramesController`, add the same optional runtime path dependency:

```python
class VideoToFramesController:
    def __init__(self, app, runtime_paths: RuntimePaths | None = None):
        self.app = app
        self.runtime_paths = runtime_paths or detect_runtime_paths()
        self.worker: VideoToFramesWorker | None = None
```

- [ ] Add config tests:

```python
def test_save_config_writes_project_ini_and_overlay(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=paths)
    video_path = str(tmp_path / "song.mp4")

    manager.save_config(video_path)

    assert paths.project_ini_path(video_path).exists()
    assert paths.project_overlay_json_path(video_path).exists()
    assert not (tmp_path / "song.ini").exists()
    assert not (tmp_path / "song_overlays.json").exists()
```

```python
def test_load_config_falls_back_to_legacy_sidecar(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    app_state = AppState()
    video_path = tmp_path / "song.mp4"
    legacy_ini = tmp_path / "song.ini"
    legacy_overlay = tmp_path / "song_overlays.json"
    legacy_ini.write_text("[calibration]\n", encoding="utf-8")
    legacy_overlay.write_text("{}", encoding="utf-8")

    manager = ConfigManager(app_state, runtime_paths=paths)

    assert legacy_ini in manager.config_candidates_for_video(str(video_path))
```

- [ ] Add frame tests:

```python
def test_frame_conversion_uses_project_frames_dir(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    video_path = tmp_path / "song.mp4"
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=paths)
    workflow = VideoLoadingWorkflow(app_state, manager, runtime_paths=paths)

    frames_dir = workflow._frames_dir_for_video(str(video_path))

    assert frames_dir == paths.project_frames_dir(str(video_path))
```

```python
def test_existing_legacy_frames_dir_is_reused(tmp_path):
    paths = RuntimePaths(
        frozen=False,
        app_root=tmp_path / "repo",
        repo_root=tmp_path / "repo",
        home_dir=tmp_path,
        platform_name="darwin",
    )
    app_state = AppState()
    manager = ConfigManager(app_state, runtime_paths=paths)
    workflow = VideoLoadingWorkflow(app_state, manager, runtime_paths=paths)
    video_path = tmp_path / "song.mp4"
    legacy_frames = tmp_path / "song_frames"
    legacy_frames.mkdir()

    assert workflow._frames_dir_for_video(str(video_path)) == legacy_frames
```

- [ ] Run:

```bash
.venv/bin/python -m pytest tests/test_config_manager.py tests/test_video_loading_paths.py tests/test_video_to_frames_controller.py
```

Expected result: project-path writes and legacy fallback tests pass.

Commit after this step:

```bash
git add synthesia2midi/synthesia2midi/config_manager.py synthesia2midi/synthesia2midi/workflows/video_loading.py synthesia2midi/synthesia2midi/workflows/video_to_frames.py tests/test_config_manager.py tests/test_video_loading_paths.py tests/test_video_to_frames_controller.py
git commit -m "Store working files in project data"
```

---

## Step 6: Move YouTube Downloads To Downloads Folder

Stop using Movies/Documents as the YouTube target default.

- [ ] Update `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py`:

```python
runtime_paths = detect_runtime_paths()
dialog = YouTubeDownloadDialog(
    self.app,
    default_output_dir=str(runtime_paths.default_download_dir()),
)
```

Leave open-video file picker behavior alone unless the current code forces downloads and video open picker to share one path.

- [ ] Keep `YouTubeDownloader._build_output_path()` behavior that creates the per-video slug folder under `output_dir`.
- [ ] Update `tests/test_packaged_entrypoint.py` so the fake runtime paths exposes `default_download_dir()` and the assertion checks that value:

```python
assert calls["default_output_dir"] == str(fake_paths.default_download_dir())
```

- [ ] Add or update `tests/test_youtube_download_dialog.py` to prove the dialog receives the dedicated download folder without including `qps` or unrelated localization behavior.
- [ ] Run:

```bash
.venv/bin/python -m pytest tests/test_youtube_download_dialog.py tests/test_packaged_entrypoint.py
```

Expected result: YouTube dialog tests pass with Downloads-based default.

Commit after this step:

```bash
git add synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py tests/test_youtube_download_dialog.py tests/test_packaged_entrypoint.py
git commit -m "Use Downloads folder for video downloads"
```

---

## Step 7: Update Localization And UI String Audit

The new visible string is `Show MIDI in Folder`. If any failure dialog is added, include those strings too.

- [ ] Regenerate the audit manifest:

```bash
.venv/bin/python -m synthesia2midi.tools.audit_ui_strings --output docs/localization/ui-string-manifest.json
```

- [ ] Regenerate Qt source catalogs:

```bash
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts
.venv/bin/pyside6-lupdate -extensions py synthesia2midi/synthesia2midi -ts synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts
```

- [ ] Add translations for `Show MIDI in Folder`:

```text
es: Mostrar MIDI en carpeta
ja: MIDI をフォルダで表示
ru: Показать MIDI в папке
zh_CN: 在文件夹中显示 MIDI
ko: 폴더에서 MIDI 보기
pt_BR: Mostrar MIDI na pasta
```

- [ ] Compile catalogs:

```bash
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_es.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_es.qm
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_ja.qm
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_ru.qm
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_zh_CN.qm
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_ko.qm
.venv/bin/pyside6-lrelease synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.ts -qm synthesia2midi/synthesia2midi/translations/synthesia2midi_pt_BR.qm
```

- [ ] Run:

```bash
.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py
```

Expected result: localization and audit tests pass; no production locale has unfinished strings or placeholder mismatches.

Commit after this step:

```bash
git add docs/localization/ui-string-manifest.json synthesia2midi/synthesia2midi/translations
git commit -m "Update localization for output directory UX"
```

---

## Step 8: Full Verification And Integration Check

- [ ] Run:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

Expected result:

```text
git diff --check exits 0
compileall exits 0
pytest exits 0
```

- [ ] Run path-specific smoke checks:

```bash
.venv/bin/python -m pytest tests/test_runtime_paths.py tests/test_config_manager.py tests/test_midi_conversion_controller.py tests/test_midi_touchup_controller.py tests/test_packaged_entrypoint.py
```

Expected result: all focused tests pass.

- [ ] Confirm no generated user data is staged:

```bash
git status --short
git diff --cached --name-only
```

Expected staged files should be source, tests, docs, localization assets, and Backlog only. There should be no generated MIDI, downloaded MP4, extracted frame JPG/PNG, or app data files.

- [ ] Confirm old fallback behavior conceptually:
  - A new save writes project `.ini` and `_overlays.json`.
  - A legacy sidecar `.ini` is still found when project config is absent.
  - A legacy `video_stem_frames/` folder is reused when project frames are absent.
  - No automatic migration moves old user files.

Final commit after all checks:

```bash
git add synthesia2midi/synthesia2midi/runtime_paths.py \
    synthesia2midi/synthesia2midi/config_manager.py \
    synthesia2midi/synthesia2midi/workflows/video_loading.py \
    synthesia2midi/synthesia2midi/workflows/video_to_frames.py \
    synthesia2midi/synthesia2midi/workflows/conversion.py \
    synthesia2midi/synthesia2midi/workflows/midi_export.py \
    synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py \
    synthesia2midi/synthesia2midi/gui/midi_conversion_controller.py \
    synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py \
    synthesia2midi/synthesia2midi/translations \
    docs/localization/ui-string-manifest.json \
    tests \
    backlog/tasks
git commit -m "Improve output directory UX"
```

Do not push.

---

## Self-Review Checklist

- [ ] The plan preserves existing user calibration/config files by reading legacy paths.
- [ ] The final MIDI output folder contains the MIDI file, not frame/config/log clutter.
- [ ] Downloaded source videos go to Downloads, not Movies.
- [ ] Working files go to platform-appropriate app data.
- [ ] The conversion-complete dialog has no silent `Done` path as a primary button.
- [ ] New user-visible strings are wrapped in translation calls and present in all production locale catalogs.
- [ ] Tests cover both new-write paths and old-read fallbacks.
- [ ] No implementation step requires real videos, network access, or visible GUI windows.
- [ ] No branch, worktree, push, or destructive git operation is part of the plan.
