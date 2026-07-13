# Startup Source Flow Design

## Summary

Make the Select Video Source dialog the only visible window when Synthesia2MIDI starts. The main workspace is constructed so existing controllers remain available, but it stays hidden until a video loads successfully.

Cancelling or closing the source selector exits the application. Cancelling a secondary file picker or YouTube dialog leaves the source selector open so the user can choose another path without seeing an empty main window.

## Goals

- Eliminate the duplicate empty main window behind the startup selector.
- Show the main workspace only after a video has loaded successfully.
- Keep the source selector open after a secondary dialog is cancelled, a downloaded video is not loaded, or loading fails.
- Exit cleanly when the user cancels or closes the source selector itself.
- Preserve the main window's File menu and empty-state actions for later in-session use.

## Non-Goals

- Removing the main window's Open Video, Download from YouTube, or Settings actions.
- Redesigning the source selector's visual layout or wording.
- Changing video loading, recent-video persistence, or YouTube download behavior after startup.
- Adding a touch-up editor action to the source selector.

## User Flow

1. Synthesia2MIDI creates its application services and main window without showing the main window.
2. The Select Video Source dialog appears as the only visible app window.
3. Open Video File opens the native file picker over the source selector.
   - A successful video load closes the selector and reveals the main workspace.
   - Cancelling the picker or failing to load keeps the selector open.
4. Download from YouTube opens the YouTube dialog over the source selector.
   - Downloading and loading a video closes the selector and reveals the main workspace.
   - Cancelling, downloading without loading, or failing keeps the selector open.
5. Selecting a recent video closes the selector only if that video loads successfully.
6. Cancelling or closing Select Video Source quits Synthesia2MIDI completely.

## Architecture

### Explicit Startup Entry Point

`Video2MidiApp` gains an explicit `begin_startup()` method. Construction no longer schedules the startup dialog by itself. Both executable entry points call `begin_startup()` and then enter the Qt event loop without calling `show()`.

This keeps ordinary tests and audit tools free to construct the main window without accidentally launching a modal dialog.

### Source Selector Ownership

`StartupDialog` remains a chooser. Its video-source buttons emit request signals but no longer call `accept()` themselves. The main-window startup coordinator invokes the existing `VideoSessionUiController` operations and accepts the selector only when the operation reports that a video loaded.

The selector remains visible behind secondary dialogs. This gives those dialogs a visible parent and naturally returns the user to the same selector when they cancel.

### Controller Results

The startup coordinator needs an explicit success result instead of inferring from a dialog return code:

- `VideoSessionUiController.open_video_file(parent=None) -> bool`
- `VideoSessionUiController.show_youtube_download_dialog(parent=None) -> bool`
- `VideoSessionUiController.open_recent_video_file(filepath) -> bool`

The optional parent is used only by the startup flow. Existing File-menu and empty-state callers omit it and retain the main window as their parent. Each method returns `True` only when a video session is loaded successfully.

### Exit Behavior

If `StartupDialog.exec()` returns rejected, the startup coordinator calls `QApplication.quit()` without showing the main window. No unsaved-work prompt is needed because no video session has loaded.

## Failure Handling

- File picker cancellation returns `False` and leaves the selector open.
- YouTube cancellation, download failure, or choosing not to load the downloaded video returns `False` and leaves the selector open.
- A missing or invalid recent video remains in the selector's existing disabled/failure handling and does not reveal the main window.
- A video loading error keeps its existing user-facing error message, then returns control to the selector.
- The application exits only when the selector itself is rejected.

## Testing

- Verify both launch paths call `begin_startup()` and do not call `show()` directly.
- Verify constructing `Video2MidiApp` does not automatically open a startup dialog.
- Verify the main window remains hidden while the selector is active.
- Verify successful local, YouTube, and recent-video loading accepts the selector and shows the main window.
- Verify local-file and YouTube cancellation leaves the selector open and the main window hidden.
- Verify selector cancellation calls `QApplication.quit()` and never shows the main window.
- Verify controller methods return accurate success values while existing menu callers remain compatible.
- Run the full Python suite, launcher smoke, pseudo-locale UI matrix, and packaged startup smoke before integration.

## Branch And Release Sequence

The touch-up editor branch was fast-forwarded into local `main` and verified before this branch was created. This startup branch will be implemented and verified locally, fast-forwarded into local `main`, and only then will the combined local `main` be pushed for GitHub Windows and macOS checks, per Jeff's requested sequence.
