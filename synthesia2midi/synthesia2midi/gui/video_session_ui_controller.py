"""Video/session UI orchestration for the Qt app."""
from __future__ import annotations

import logging
import os

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QDialog, QFileDialog

from synthesia2midi.gui.youtube_download_dialog import YouTubeDownloadDialog
from synthesia2midi.runtime_paths import detect_runtime_paths

translate = QCoreApplication.translate


class VideoSessionUiController:
    """Owns user-facing video/session file dialogs and frame-range handlers."""

    def __init__(self, app):
        self.app = app

    def show_youtube_download_dialog(self) -> None:
        app = self.app
        logging.info("_show_youtube_download_dialog: Showing YouTube download dialog.")
        download_dir = str(detect_runtime_paths().default_download_dir())
        dialog = YouTubeDownloadDialog(app, default_output_dir=download_dir)
        dialog.video_downloaded.connect(self.handle_youtube_video_downloaded)

        if dialog.exec() != QDialog.Accepted:
            logging.info("_show_youtube_download_dialog: User cancelled YouTube dialog, continuing with empty application.")

    def open_video_file(self) -> None:
        app = self.app
        logging.info("_open_video_file: Method started.")
        dialog = QFileDialog(app)
        dialog.setWindowTitle(translate("VideoSessionUiController", "Open Video File"))
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(
            translate(
                "VideoSessionUiController",
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)",
            )
        )
        dialog.setDirectory(str(detect_runtime_paths().default_video_dir()))

        if dialog.exec() == QDialog.Accepted:
            selected_paths = dialog.selectedFiles()
            if selected_paths:
                filepath = selected_paths[0]
                logging.info("_open_video_file: User selected %s", filepath)
                loaded = app.video_session_coordinator.load_path(
                    filepath,
                    log_prefix="_open_video_file",
                    update_fps_display=True,
                )
                if loaded:
                    self._record_recent_video(filepath)
            return

        logging.info("_open_video_file: User cancelled file dialog, continuing with empty application.")

    def open_image_sequence_folder(self) -> None:
        """Open a directory picker for an extracted frame sequence."""
        filepath = QFileDialog.getExistingDirectory(
            self.app,
            translate("VideoSessionUiController", "Open Image Sequence Folder"),
            str(detect_runtime_paths().default_video_dir()),
        )
        if not filepath:
            logging.info("Image sequence folder selection cancelled")
            return

        loaded = self.app.video_session_coordinator.load_path(
            filepath,
            log_prefix="_open_image_sequence_folder",
            update_fps_display=True,
        )
        if loaded:
            self._record_recent_video(filepath)

    def open_recent_video_file(self, filepath: str) -> None:
        logging.info("_open_recent_video_file: Loading recent video %s", filepath)
        loaded = self.app.video_session_coordinator.load_path(
            filepath,
            log_prefix="_open_recent_video_file",
            update_fps_display=True,
        )
        if loaded:
            self._record_recent_video(filepath)

    def handle_youtube_video_downloaded(self, filepath: str) -> None:
        logging.info("_handle_youtube_video_downloaded: Video downloaded to %s", filepath)
        self.app.video_session_coordinator.load_path(
            filepath,
            log_prefix="_handle_youtube_video_downloaded",
            update_fps_display=False,
        )

    def _record_recent_video(self, filepath: str) -> None:
        recent_video_store = getattr(self.app, "recent_video_store", None)
        if recent_video_store is not None:
            recent_video_store.add(filepath)

    def handle_video_to_frames_request(self) -> None:
        return self.app.video_to_frames_controller.handle_request()

    def update_nav_interval(self, value: int) -> None:
        app = self.app
        app.parameter_manager.update_nav_interval(value)
        if hasattr(app, "frame_nav_actions"):
            for nav_interval, action in app.frame_nav_actions.items():
                action.setChecked(nav_interval == value)

    def handle_frame_nav_interval(self, interval: int) -> None:
        app = self.app
        app.app_state.video.current_nav_interval = interval
        app.app_state.unsaved_changes = True

        for nav_interval, action in app.frame_nav_actions.items():
            action.setChecked(nav_interval == interval)

        if hasattr(app.control_panel, "nav_interval_changed"):
            app.control_panel.nav_interval_changed.emit(interval)

        logging.info("Frame navigation interval changed to: %s", interval)

    def handle_start_frame_change(self, frame: int) -> None:
        video = self.app.app_state.video
        video.start_frame = frame
        video.processing_start_frame = frame
        self.app.app_state.unsaved_changes = True

    def handle_end_frame_change(self, frame: int) -> None:
        video = self.app.app_state.video
        video.end_frame = frame
        video.processing_end_frame = frame
        self.app.app_state.unsaved_changes = True

    def handle_processing_start_frame_change(self, frame_value: int) -> None:
        app = self.app
        video = app.app_state.video

        if video.video_is_trimmed:
            min_frame = video.trim_start_frame
            max_frame = video.trim_end_frame
            frame_value = max(min_frame, min(frame_value, max_frame))
        else:
            total_frames = getattr(video, "total_frames", 0)
            if total_frames > 0:
                frame_value = max(0, min(frame_value, total_frames - 1))

        if video.processing_end_frame > 0 and frame_value >= video.processing_end_frame:
            logging.warning(
                "Processing start frame %s must be less than end frame %s",
                frame_value,
                video.processing_end_frame,
            )
            return

        video.processing_start_frame = frame_value
        app.app_state.mark_unsaved()
        logging.info("Set MIDI processing start frame to: %s", frame_value)

    def handle_processing_end_frame_change(self, frame_value: int) -> None:
        app = self.app
        video = app.app_state.video

        if video.video_is_trimmed:
            min_frame = video.trim_start_frame
            max_frame = video.trim_end_frame
            frame_value = max(min_frame, min(frame_value, max_frame))
        else:
            total_frames = getattr(video, "total_frames", 0)
            if total_frames > 0:
                frame_value = max(0, min(frame_value, total_frames - 1))

        if video.processing_start_frame > 0 and frame_value <= video.processing_start_frame:
            logging.warning(
                "Processing end frame %s must be greater than start frame %s",
                frame_value,
                video.processing_start_frame,
            )
            return

        video.processing_end_frame = frame_value
        app.app_state.mark_unsaved()
        logging.info("Set MIDI processing end frame to: %s", frame_value)

    def handle_trim_video_request(self, start_frame: int, end_frame: int) -> None:
        app = self.app
        video = app.app_state.video

        video.trim_start_frame = start_frame
        video.trim_end_frame = end_frame if end_frame != -1 else video.total_frames - 1
        video.video_is_trimmed = True
        video.processing_start_frame = video.trim_start_frame
        video.processing_end_frame = video.trim_end_frame

        app.control_panel.update_controls_from_state()
        app.control_panel.update_video_frame_limits()
        app.video_controls.update_frame_slider_for_video()
        app.video_controls.display_frame_with_slider_update(start_frame)
        app.app_state.mark_unsaved()

        if app.video_loading_workflow:
            success = app.video_loading_workflow.save_current_config()
            if success:
                logging.info("Video trim settings automatically saved to config file.")
            else:
                logging.warning("Auto-save of video trim settings failed.")

        logging.info(
            "Video trimmed to frames %s to %s. MIDI processing range updated accordingly.",
            start_frame,
            video.trim_end_frame,
        )

    def initialize_processing_range_defaults(self) -> None:
        app = self.app
        video = app.app_state.video

        if video.processing_start_frame == 0 and video.processing_end_frame == 0:
            if video.video_is_trimmed and video.trim_start_frame > 0:
                video.processing_start_frame = video.trim_start_frame
                video.processing_end_frame = video.trim_end_frame if video.trim_end_frame > 0 else video.total_frames - 1
                logging.info(
                    "Set processing range defaults from trim range: %s to %s",
                    video.processing_start_frame,
                    video.processing_end_frame,
                )
            else:
                video.processing_start_frame = 0
                video.processing_end_frame = video.total_frames - 1 if video.total_frames > 0 else 0
                logging.info(
                    "Set processing range defaults to full video: %s to %s",
                    video.processing_start_frame,
                    video.processing_end_frame,
                )

            if hasattr(app.control_panel, "processing_start_frame_spin"):
                app.control_panel.processing_start_frame_spin.setValue(video.processing_start_frame)
                app.control_panel.processing_end_frame_spin.setValue(video.processing_end_frame)
