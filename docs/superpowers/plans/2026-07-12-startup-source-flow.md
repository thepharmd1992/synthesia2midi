# Startup Source Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Select Video Source the only visible startup window, reveal the main workspace only after a successful video load, and exit when the selector itself is cancelled.

**Architecture:** `Video2MidiApp.begin_startup()` explicitly schedules a modal startup coordinator while the main window remains hidden. `StartupDialog` emits requests without closing itself, and `VideoSessionUiController` returns explicit success booleans so the coordinator accepts the selector only after a video session loads. Both executable entry points begin this flow without calling `show()` directly.

**Tech Stack:** Python 3.12, PySide6/Qt 6, pytest, Qt offscreen platform, PyInstaller.

## Global Constraints

- Work on `codex/startup-source-flow` in the existing checkout; do not create a worktree.
- Do not push this branch or local `main` until Jeff explicitly requests the combined CI push.
- Keep the main workspace hidden until a video loads successfully.
- Cancelling or closing Select Video Source exits Synthesia2MIDI.
- Cancelling a local-file or YouTube secondary dialog leaves the same source selector open.
- Preserve the main window's File menu and empty-state actions for in-session use.
- Do not add a touch-up editor action to the source selector.
- Do not add or change user-visible strings; localization catalogs and the UI-string manifest should remain stable.
- Preserve recent-video persistence and existing video-loading orchestration.
- Do not stage or modify the unrelated untracked `uv.lock`.
- After all local gates pass, fast-forward this branch into local `main`, verify the merged result, delete the merged local branch, and leave local `main` unpushed for the combined GitHub checks.

---

## File Map

**Startup result contracts**

- Modify `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py`: accept an optional visible dialog parent and return whether a video loaded.
- Modify `tests/test_recent_videos.py`: cover local/recent success, failure, cancellation, and parent selection.
- Modify `tests/test_packaged_entrypoint.py`: cover YouTube cancellation, parent selection, and successful download/load return values.

**Selector and main-window coordination**

- Modify `synthesia2midi/synthesia2midi/gui/startup_dialog.py`: emit source requests without accepting the dialog.
- Modify `synthesia2midi/synthesia2midi/main.py`: remove constructor-driven startup, add `begin_startup()`, coordinate request success, quit on selector rejection, and show only after load.
- Modify `tests/test_startup_dialog.py` and `tests/test_recent_videos.py`: verify source requests do not close the selector.
- Modify `tests/test_video2midi_app_smoke.py`: verify scheduling, hidden state, success, secondary cancellation, and selector cancellation.

**Executable entry points**

- Modify `synthesia2midi/run.py`: call `begin_startup()` instead of `show()`.
- Modify the direct-execution guard in `synthesia2midi/synthesia2midi/main.py`: call `begin_startup()` instead of `show()`.
- Modify `tests/test_packaged_entrypoint.py`: inspect both launch guards and prevent direct main-window showing.

**Tracking and final gates**

- Modify `backlog/tasks/task-24 - Make-source-selector-the-only-startup-window.md`: mark progress, record verification, and complete acceptance criteria.
- Verify `docs/localization/ui-string-manifest.json` and translation catalogs have no diff.

---

### Task 1: Return Explicit Video-Load Results

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py:16-103`
- Test: `tests/test_recent_videos.py:145-263`
- Test: `tests/test_packaged_entrypoint.py:51-83`
- Modify: `backlog/tasks/task-24 - Make-source-selector-the-only-startup-window.md:1-40`

**Interfaces:**
- Consumes: `VideoSessionCoordinator.load_path(...) -> bool`, `YouTubeDownloadDialog.video_downloaded`, and optional `QWidget` parents.
- Produces: `open_video_file(parent: QWidget | None = None) -> bool`, `show_youtube_download_dialog(parent: QWidget | None = None) -> bool`, `open_recent_video_file(filepath: str) -> bool`, and `handle_youtube_video_downloaded(filepath: str) -> bool`.

- [ ] **Step 1: Mark TASK-24 In Progress**

Change only the task front matter:

```yaml
status: In Progress
```

- [ ] **Step 2: Write failing local-file and recent-video result tests**

In `test_open_video_file_records_recent_only_after_success`, add a `parent_marker`, record the constructed dialog, and assert the success result and parent:

```python
parent_marker = object()
created_dialogs = []

class FakeFileDialog:
    ExistingFile = object()

    def __init__(self, parent):
        self.parent = parent
        created_dialogs.append(self)

    def setWindowTitle(self, value):
        pass

    def setFileMode(self, value):
        pass

    def setNameFilter(self, value):
        pass

    def setDirectory(self, value):
        pass

    def exec(self):
        return QDialog.Accepted

    def selectedFiles(self):
        return [selected_path]

result = VideoSessionUiController(app).open_video_file(parent=parent_marker)

assert result is True
assert created_dialogs[0].parent is parent_marker
assert recent_store.paths == [selected_path]
```

```python
result = VideoSessionUiController(app).open_video_file()

assert result is False
assert recent_store.paths == []
```

Add a rejected file-dialog test:

```python
def test_open_video_file_returns_false_when_picker_is_cancelled(monkeypatch):
    from synthesia2midi.gui import video_session_ui_controller as module

    class RejectedFileDialog:
        ExistingFile = object()

        def __init__(self, parent):
            self.parent = parent

        def setWindowTitle(self, value):
            pass

        def setFileMode(self, value):
            pass

        def setNameFilter(self, value):
            pass

        def setDirectory(self, value):
            pass

        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(module, "QFileDialog", RejectedFileDialog)
    app = SimpleNamespace(recent_video_store=RecordingRecentStore())

    assert VideoSessionUiController(app).open_video_file() is False
```

Update the recent-video test to retain the return value:

```python
assert controller.open_recent_video_file(recent_path) is True
assert recent_store.paths == [recent_path]
```

Add a failed recent load assertion:

```python
app.video_session_coordinator.load_path = lambda *args, **kwargs: False
assert controller.open_recent_video_file(recent_path) is False
```

- [ ] **Step 3: Write failing YouTube result and parent tests**

Extend `tests/test_packaged_entrypoint.py`:

```python
def test_youtube_dialog_returns_false_on_cancel_and_uses_requested_parent(monkeypatch, tmp_path):
    from synthesia2midi.gui import video_session_ui_controller as module

    calls = {}
    parent_marker = object()

    class Signal:
        def connect(self, callback):
            calls["callback"] = callback

    class FakeDialog:
        def __init__(self, parent=None, default_output_dir=""):
            calls["parent"] = parent
            self.video_downloaded = Signal()

        def exec(self):
            return QDialog.Rejected

    monkeypatch.setattr(module, "YouTubeDownloadDialog", FakeDialog)
    controller = module.VideoSessionUiController(app=object())

    assert controller.show_youtube_download_dialog(parent=parent_marker) is False
    assert calls["parent"] is parent_marker
```

Add a successful signal/load test using a coordinator whose `load_path` returns `True`:

```python
def exec(self):
    calls["callback"]("/tmp/downloaded.mp4")
    return QDialog.Accepted

assert controller.show_youtube_download_dialog() is True
```

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_recent_videos.py tests/test_packaged_entrypoint.py -q
```

Expected: failures show unexpected keyword argument `parent` and `None is not True/False` because the controller methods do not yet expose result contracts.

- [ ] **Step 5: Implement explicit results and optional parents**

In `video_session_ui_controller.py`, import `QWidget` and update the methods:

```python
def show_youtube_download_dialog(self, parent: QWidget | None = None) -> bool:
    app = self.app
    dialog_parent = app if parent is None else parent
    download_dir = str(detect_runtime_paths().default_download_dir())
    dialog = YouTubeDownloadDialog(dialog_parent, default_output_dir=download_dir)
    loaded = False

    def handle_downloaded(filepath: str) -> None:
        nonlocal loaded
        loaded = self.handle_youtube_video_downloaded(filepath)

    dialog.video_downloaded.connect(handle_downloaded)
    dialog.exec()
    return loaded
```

```python
def open_video_file(self, parent: QWidget | None = None) -> bool:
    app = self.app
    dialog_parent = app if parent is None else parent
    dialog = QFileDialog(dialog_parent)
    dialog.setWindowTitle(translate("VideoSessionUiController", "Open Video File"))
    dialog.setFileMode(QFileDialog.ExistingFile)
    dialog.setNameFilter(
        translate(
            "VideoSessionUiController",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
        )
    )
    dialog.setDirectory(str(detect_runtime_paths().default_video_dir()))
    if dialog.exec() != QDialog.Accepted:
        logging.info("_open_video_file: User cancelled file dialog.")
        return False
    selected_paths = dialog.selectedFiles()
    if not selected_paths:
        return False
    filepath = selected_paths[0]
    loaded = app.video_session_coordinator.load_path(
        filepath,
        log_prefix="_open_video_file",
        update_fps_display=True,
    )
    if loaded:
        self._record_recent_video(filepath)
    return bool(loaded)
```

```python
def open_recent_video_file(self, filepath: str) -> bool:
    loaded = self.app.video_session_coordinator.load_path(
        filepath,
        log_prefix="_open_recent_video_file",
        update_fps_display=True,
    )
    if loaded:
        self._record_recent_video(filepath)
    return bool(loaded)

def handle_youtube_video_downloaded(self, filepath: str) -> bool:
    return bool(
        self.app.video_session_coordinator.load_path(
            filepath,
            log_prefix="_handle_youtube_video_downloaded",
            update_fps_display=False,
        )
    )
```

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the Step 4 command.

Expected: all selected tests pass, including existing recent-video and packaged-path behavior.

- [ ] **Step 7: Commit the controller contract checkpoint**

```bash
git add \
  'backlog/tasks/task-24 - Make-source-selector-the-only-startup-window.md' \
  synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py \
  tests/test_recent_videos.py \
  tests/test_packaged_entrypoint.py
git commit -m "refactor: report video source load results"
```

---

### Task 2: Keep the Selector Open Until Loading Succeeds

**Files:**
- Modify: `synthesia2midi/synthesia2midi/gui/startup_dialog.py:179-243`
- Modify: `synthesia2midi/synthesia2midi/main.py:75-140,490-503`
- Test: `tests/test_startup_dialog.py`
- Test: `tests/test_recent_videos.py:62-78`
- Test: `tests/test_video2midi_app_smoke.py:13-103`

**Interfaces:**
- Consumes: Task 1's controller methods returning `bool`.
- Produces: `Video2MidiApp.begin_startup() -> None`, `_finish_startup_action(dialog: QDialog, loaded: bool) -> None`, and a startup dialog that closes only when the coordinator calls `accept()`.

- [ ] **Step 1: Write failing selector-request tests**

In `tests/test_startup_dialog.py`, add:

```python
def test_startup_source_buttons_request_actions_without_closing_dialog():
    QApplication.instance() or QApplication([])
    dialog = StartupDialog()
    requests = []
    finished = []
    dialog.open_local_file.connect(lambda: requests.append("local"))
    dialog.download_from_youtube.connect(lambda: requests.append("youtube"))
    dialog.finished.connect(finished.append)

    dialog.local_file_btn.click()
    dialog.youtube_btn.click()

    assert requests == ["local", "youtube"]
    assert finished == []
```

Replace `test_startup_dialog_emits_recent_file_before_closing` in `tests/test_recent_videos.py` with:

```python
def test_startup_dialog_recent_file_request_does_not_close_selector(tmp_path):
    QApplication.instance() or QApplication([])
    recent_path = tmp_path / "song.mp4"
    recent_path.write_text("video")
    dialog = StartupDialog(recent_video_paths=[str(recent_path)])
    emitted = []
    finished = []
    dialog.open_recent_file.connect(emitted.append)
    dialog.finished.connect(finished.append)

    dialog.recent_video_buttons[0].click()

    assert emitted == [str(recent_path)]
    assert finished == []
```

- [ ] **Step 2: Write failing main-window startup coordinator tests**

Update the test doubles in `tests/test_video2midi_app_smoke.py`:

```python
class FakeStartupSignal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self, *args):
        for slot in self._slots:
            slot(*args)


class FakeStartupDialog:
    action = "reject"
    instances = []

    def __init__(self, parent, *, recent_video_paths=None):
        self.parent = parent
        self.recent_video_paths = list(recent_video_paths or [])
        self.open_local_file = FakeStartupSignal()
        self.open_recent_file = FakeStartupSignal()
        self.download_from_youtube = FakeStartupSignal()
        self.accepted = False
        FakeStartupDialog.instances.append(self)

    def accept(self):
        self.accepted = True

    def reject(self):
        self.accepted = False

    def exec(self):
        if self.action == "local":
            self.open_local_file.emit()
        elif self.action == "youtube":
            self.download_from_youtube.emit()
        elif isinstance(self.action, tuple) and self.action[0] == "recent":
            self.open_recent_file.emit(self.action[1])
        return QDialog.Accepted if self.accepted else QDialog.Rejected
```

Add a scheduling test:

```python
def test_begin_startup_schedules_selector_without_showing_main(monkeypatch):
    scheduled = []
    monkeypatch.setattr(QTimer, "singleShot", lambda delay, callback: scheduled.append((delay, callback)))
    qt_app = QApplication.instance() or QApplication([])
    window = Video2MidiApp()

    try:
        assert scheduled == []
        assert not window.isVisible()

        window.begin_startup()

        assert len(scheduled) == 1
        assert scheduled[0][0] == 0
        assert scheduled[0][1] == window._show_startup_dialog
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
        qt_app.processEvents()
```

Add a successful local load test:

```python
def test_startup_local_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "local"
    quit_calls = []
    monkeypatch.setattr(main_module.QApplication, "quit", lambda *args: quit_calls.append("quit"))

    def load_local(self, parent=None):
        assert parent is FakeStartupDialog.instances[-1]
        self.app.video_session = object()
        return True

    monkeypatch.setattr(VideoSessionUiController, "open_video_file", load_local)
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
        assert quit_calls == []
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()

```

Add a secondary-cancellation test:

```python
def test_startup_secondary_cancel_keeps_main_hidden_until_selector_rejects(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "local"
    quit_calls = []
    monkeypatch.setattr(main_module.QApplication, "quit", lambda *args: quit_calls.append("quit"))
    monkeypatch.setattr(
        VideoSessionUiController,
        "open_video_file",
        lambda self, parent=None: False,
    )
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert not FakeStartupDialog.instances[-1].accepted
        assert not window.isVisible()
        assert quit_calls == ["quit"]
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()

```

Add a YouTube success test whose fake dialog emits the download request and whose controller returns `True`:

```python
def test_startup_youtube_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = "youtube"

    def load_youtube(self, parent=None):
        assert parent is FakeStartupDialog.instances[-1]
        self.app.video_session = object()
        return True

    monkeypatch.setattr(
        VideoSessionUiController,
        "show_youtube_download_dialog",
        load_youtube,
    )
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
```

Add a recent-video success test whose fake emits a concrete path and whose controller returns `True`:

```python
def test_startup_recent_success_accepts_selector_and_shows_main(monkeypatch):
    monkeypatch.setattr(QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(main_module, "StartupDialog", FakeStartupDialog)
    FakeStartupDialog.instances.clear()
    FakeStartupDialog.action = ("recent", "/tmp/recent.mp4")

    def load_recent(self, filepath):
        assert filepath == "/tmp/recent.mp4"
        self.app.video_session = object()
        return True

    monkeypatch.setattr(
        VideoSessionUiController,
        "open_recent_video_file",
        load_recent,
    )
    window = Video2MidiApp()

    try:
        window._show_startup_dialog()

        assert FakeStartupDialog.instances[-1].accepted
        assert window.isVisible()
    finally:
        window.app_state.unsaved_changes = False
        window.close()
        window.deleteLater()
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_startup_dialog.py \
  tests/test_recent_videos.py \
  tests/test_video2midi_app_smoke.py -q
```

Expected: selector request tests report unexpected `finished` emissions, and startup tests fail because construction still schedules the dialog and `begin_startup` does not exist.

- [ ] **Step 4: Stop source buttons from accepting themselves**

In `startup_dialog.py`:

```python
def _on_local_file_clicked(self):
    self.open_local_file.emit()

def _on_youtube_clicked(self):
    self.download_from_youtube.emit()

def _on_recent_file_clicked(self, path: str):
    self.open_recent_file.emit(path)
```

Keep the Cancel button connected to `reject()`.

- [ ] **Step 5: Add explicit startup scheduling and coordination**

Remove this constructor behavior from `Video2MidiApp.__init__`:

```python
QTimer.singleShot(100, self._show_startup_dialog)
```

Add:

```python
def begin_startup(self) -> None:
    """Start the source-selection flow without revealing the main workspace."""
    QTimer.singleShot(0, self._show_startup_dialog)

@staticmethod
def _finish_startup_action(dialog: QDialog, loaded: bool) -> None:
    if loaded:
        dialog.accept()
```

Replace `_show_startup_dialog` with a defensive loop:

```python
def _show_startup_dialog(self) -> None:
    while not self.has_video_loaded():
        dialog = StartupDialog(
            self,
            recent_video_paths=self.recent_video_store.recent_paths(),
        )
        dialog.open_local_file.connect(
            lambda: self._finish_startup_action(
                dialog,
                self.video_session_ui_controller.open_video_file(parent=dialog),
            )
        )
        dialog.download_from_youtube.connect(
            lambda: self._finish_startup_action(
                dialog,
                self.video_session_ui_controller.show_youtube_download_dialog(parent=dialog),
            )
        )
        dialog.open_recent_file.connect(
            lambda path: self._finish_startup_action(
                dialog,
                self.video_session_ui_controller.open_recent_video_file(path),
            )
        )

        if dialog.exec() != QDialog.Accepted:
            logging.info("Startup source selector cancelled; exiting application.")
            qapp = QApplication.instance()
            if qapp is not None:
                qapp.quit()
            return

    self.show()
```

The loop handles an unexpected accepted-without-load result by showing a fresh selector rather than exposing an empty workspace.

- [ ] **Step 6: Run the focused tests and verify GREEN**

Run the Step 3 command.

Expected: all selected tests pass; the main window is visible only in the successful-load cases.

- [ ] **Step 7: Commit the startup coordination checkpoint**

```bash
git add \
  synthesia2midi/synthesia2midi/gui/startup_dialog.py \
  synthesia2midi/synthesia2midi/main.py \
  tests/test_startup_dialog.py \
  tests/test_recent_videos.py \
  tests/test_video2midi_app_smoke.py
git commit -m "feat: keep the workspace hidden during startup"
```

---

### Task 3: Route Both Executable Entry Points Through Hidden Startup

**Files:**
- Modify: `synthesia2midi/run.py:114-151`
- Modify: `synthesia2midi/synthesia2midi/main.py:677-681`
- Test: `tests/test_packaged_entrypoint.py`
- Test: `tests/test_import_smoke.py`

**Interfaces:**
- Consumes: `Video2MidiApp.begin_startup() -> None` from Task 2.
- Produces: both executable entry points enter the Qt event loop without directly showing the main window.

- [ ] **Step 1: Write a failing AST-based launch-guard test**

Add to `tests/test_packaged_entrypoint.py`:

```python
import ast


def _main_guard_calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(part, ast.Constant) and part.value == "__main__"
            for part in ast.walk(node.test)
        )
    )
    return {
        ast.unparse(node.func)
        for node in ast.walk(guard)
        if isinstance(node, ast.Call)
    }


def test_gui_launch_guards_begin_startup_without_showing_main():
    paths = [
        ROOT / "synthesia2midi" / "run.py",
        ROOT / "synthesia2midi" / "synthesia2midi" / "main.py",
    ]
    for path in paths:
        calls = _main_guard_calls(path)
        assert "app.begin_startup" in calls
        assert "app.show" not in calls
```

- [ ] **Step 2: Run the launch test and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_packaged_entrypoint.py::test_gui_launch_guards_begin_startup_without_showing_main -q
```

Expected: failure because both guards contain `app.show` and neither contains `app.begin_startup`.

- [ ] **Step 3: Replace direct showing in both entry points**

In `synthesia2midi/run.py`:

```python
logger.info("Starting source-selection flow...")
app.begin_startup()
logger.info("Source-selection flow scheduled")

logger.info("Starting Qt event loop...")
exit_code = qapp.exec()
```

Remove the `app.show()` call and its "Application window shown" log.

In the direct guard at the bottom of `synthesia2midi/synthesia2midi/main.py`:

```python
qapp = QApplication(sys.argv)
app = Video2MidiApp()
app.begin_startup()
sys.exit(qapp.exec())
```

- [ ] **Step 4: Run launch, import, and packaged-entrypoint tests**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_packaged_entrypoint.py \
  tests/test_import_smoke.py \
  tests/test_video2midi_app_smoke.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the executable-entrypoint checkpoint**

```bash
git add \
  synthesia2midi/run.py \
  synthesia2midi/synthesia2midi/main.py \
  tests/test_packaged_entrypoint.py
git commit -m "fix: launch with only the source selector visible"
```

---

### Task 4: Verify the Real Startup Experience and Close TASK-24

**Files:**
- Modify: `backlog/tasks/task-24 - Make-source-selector-the-only-startup-window.md`
- Verify unchanged: `docs/localization/ui-string-manifest.json`
- Verify unchanged: `synthesia2midi/synthesia2midi/translations/`

**Interfaces:**
- Consumes: completed source-selection startup flow from Tasks 1-3.
- Produces: local verification evidence and a branch ready for the already-authorized fast-forward merge into local `main`.

- [ ] **Step 1: Run the focused startup regression gate**

```bash
.venv/bin/python -m pytest \
  tests/test_startup_dialog.py \
  tests/test_recent_videos.py \
  tests/test_video2midi_app_smoke.py \
  tests/test_video_session_load_ordering.py \
  tests/test_packaged_entrypoint.py \
  tests/test_import_smoke.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run source verification**

```bash
git diff --check main..HEAD
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m ruff check synthesia2midi tests --select=E9,F63,F7,F82
.venv/bin/python -m pytest
cargo fmt --manifest-path tools/midi_touchup_editor_rust/Cargo.toml --check
cargo test --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

Expected: no diff/compile/lint failures; all Python and Rust tests pass. Existing Qt deprecation and unused `AudioTelemetry::meter` warnings may remain.

- [ ] **Step 3: Render the pseudo-locale UI matrix**

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m synthesia2midi.tools.render_ui_matrix \
  --locale qps \
  --font-scale 1.5 \
  --output logs/ux-audit/startup-source-flow
```

Expected: `report.json` lists all surfaces as nonblank with no clipping. The ignored `logs/` output is not staged.

- [ ] **Step 4: Smoke the real developer launcher on macOS**

Launch `Launch Synthesia2MIDI.command` and verify visually:

- Select Video Source is the only visible Synthesia2MIDI window.
- Cancelling the native file picker returns to the same selector.
- Cancelling the selector closes the Python/Synthesia2MIDI process.

Use Computer Use or the visible app session for this check; do not rely only on pytest.

- [ ] **Step 5: Build and smoke the Apple Silicon package**

```bash
.venv/bin/python packaging/build_release.py --version v0.2.1-dev
```

Expected: `dist/release/Synthesia2MIDI-macos-arm64-v0.2.1-dev.zip` is rebuilt, and the package smoke remains alive for the build script's eight-second check while the selector is visible offscreen.

- [ ] **Step 6: Confirm localization artifacts are unchanged**

```bash
git diff --name-only main..HEAD -- docs/localization synthesia2midi/synthesia2midi/translations
```

Expected: no output because this feature adds no user-visible text.

- [ ] **Step 7: Update TASK-24 with exact evidence**

Mark acceptance criteria complete, set `status: Done`, and record:

- Focused and full Python test counts.
- Rust test/check result.
- UI matrix surface/clipping result.
- Developer-launcher visual result.
- Package archive name and smoke result.
- Confirmation that no localization assets changed.

- [ ] **Step 8: Commit the verification record**

```bash
git add 'backlog/tasks/task-24 - Make-source-selector-the-only-startup-window.md'
git commit -m "docs: record startup source flow verification"
```

- [ ] **Step 9: Request an independent code review**

Review `main..HEAD` against:

- `docs/superpowers/specs/2026-07-12-startup-source-flow-design.md`
- This implementation plan.
- TASK-24 acceptance criteria.

Fix verified Critical or Important findings with a failing regression test first, then rerun Steps 1-6 and update the verification record if counts change.

---

## Local Integration Sequence

Jeff has already selected local integration before the combined push. After Task 4 and review complete:

1. Confirm `git status --short --branch` shows only the unrelated untracked `uv.lock`.
2. Check out local `main`.
3. Fast-forward only: `git merge --ff-only codex/startup-source-flow`.
4. Rerun the focused startup regression gate and full Python suite on merged `main`.
5. Delete the merged local branch with `git branch -d codex/startup-source-flow`.
6. Do not push. Report that local `main` now contains both touch-up and startup-flow work and awaits Jeff's explicit combined push authorization.

## Deferred Combined Remote Gate

When Jeff authorizes the push:

1. Push local `main` to `origin/main`.
2. Wait for GitHub Python and Rust matrices on Windows, macOS, and Linux.
3. Use `superpowers:systematic-debugging` for any failure; fix on a new branch, verify, merge locally, and push only under the active authorization.
4. Run the tag-triggered Windows x64 and Apple Silicon package matrix when Jeff authorizes the release version.
