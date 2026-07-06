"""Video-to-frame-series conversion controller.

This module keeps the FFmpeg frame extraction workflow out of the main Qt window
while preserving the existing UI behavior and signal wiring.
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess

from PySide6.QtCore import QCoreApplication, QThread, Signal
from PySide6.QtWidgets import QMessageBox

translate = QCoreApplication.translate


class VideoToFramesWorker(QThread):
    """Worker thread for video to frames conversion to avoid blocking the GUI."""

    progress_updated = Signal(str)
    conversion_finished = Signal(bool, str)

    def __init__(self, video_path: str, output_dir: str, quality: int = 90):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.quality = quality

    def run(self):
        """Run the video conversion in a separate thread."""
        try:
            self.progress_updated.emit("Starting video to frame series conversion...")
            os.makedirs(self.output_dir, exist_ok=True)

            from synthesia2midi.utils.ffmpeg_helper import find_ffmpeg

            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                self.conversion_finished.emit(False, "FFmpeg not found. Please install FFmpeg.")
                return

            output_pattern = os.path.join(self.output_dir, "frame_%06d.jpg")
            cmd = [
                ffmpeg_path,
                "-y",
                "-i",
                self.video_path,
                "-q:v",
                str(100 - self.quality),
                "-vf",
                "format=bgr24",
                output_pattern,
            ]

            self.progress_updated.emit("Running ffmpeg conversion...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                frame_count = len(glob.glob(os.path.join(self.output_dir, "frame_*.jpg")))
                success_msg = f"Successfully converted {frame_count} frames to {self.output_dir}"
                self.progress_updated.emit(success_msg)
                self.conversion_finished.emit(True, success_msg)
            else:
                stderr = result.stderr.strip()
                if "Is a directory" in stderr:
                    error_msg = "Error: Input path is a directory, not a video file. Please load the original video file."
                elif "No such file or directory" in stderr:
                    error_msg = "Error: Video file not found. Please check the file path."
                elif stderr:
                    stderr_lines = stderr.split("\n")
                    relevant_lines = [
                        line for line in stderr_lines[-10:] if line.strip() and not line.startswith("[")
                    ]
                    error_msg = f"FFmpeg conversion failed:\n{chr(10).join(relevant_lines)}"
                else:
                    error_msg = f"FFmpeg conversion failed with return code: {result.returncode}"
                self.conversion_finished.emit(False, error_msg)

        except Exception as exc:
            self.conversion_finished.emit(False, f"Error during conversion: {exc}")


class VideoToFramesController:
    """Coordinates video-to-frame conversion UI flow for the main window."""

    def __init__(self, app):
        self.app = app
        self.worker: VideoToFramesWorker | None = None

    def handle_request(self):
        """Handle request to convert current video to frame series."""
        app = self.app
        if not app.app_state.video.filepath:
            QMessageBox.warning(
                app,
                translate("VideoToFramesController", "Video to Frames"),
                translate("VideoToFramesController", "No video file is open. Open a video first."),
            )
            return

        from synthesia2midi.utils.ffmpeg_helper import check_ffmpeg_available

        is_available, message = check_ffmpeg_available()
        if not is_available:
            QMessageBox.critical(
                app,
                translate("VideoToFramesController", "FFmpeg Not Found"),
                translate(
                    "VideoToFramesController",
                    "{message}\n\nPlease install FFmpeg:\n• Windows: Download from https://ffmpeg.org/download.html\n• macOS: brew install ffmpeg\n• Linux: sudo apt install ffmpeg",
                ).format(message=message),
            )
            return

        video_path = app.app_state.video.filepath

        if os.path.isdir(video_path):
            if video_path.endswith("_frames"):
                base_path = video_path[:-7]
                parent_dir = os.path.dirname(base_path)
                base_name = os.path.basename(base_path)
                video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]
                original_video = None

                for ext in video_extensions:
                    candidate_path = base_path + ext
                    if os.path.isfile(candidate_path):
                        original_video = candidate_path
                        break

                if original_video:
                    video_path = original_video
                    QMessageBox.information(
                        app,
                        translate("VideoToFramesController", "Video to Frames"),
                        translate(
                            "VideoToFramesController",
                            "Frame series is currently loaded. Found original video file:\n\n{video_name}\n\nWill convert this video to update the frame series.",
                        ).format(video_name=os.path.basename(original_video)),
                    )
                else:
                    QMessageBox.warning(
                        app,
                        translate("VideoToFramesController", "Video to Frames"),
                        translate(
                            "VideoToFramesController",
                            "A frame series is currently loaded, but the original video file could not be found.\n\nFrame series path: {video_path}\nExpected video in: {parent_dir}/\nWith name: {base_name}.mp4 (or .mov, .avi, etc.)\n\nPlease load the original video file manually.",
                        ).format(video_path=video_path, parent_dir=parent_dir, base_name=base_name),
                    )
                    return
            else:
                QMessageBox.warning(
                    app,
                    translate("VideoToFramesController", "Video to Frames"),
                    translate(
                        "VideoToFramesController",
                        "A directory is currently loaded, but it doesn't appear to be a frame series.\n\nCurrent path: {video_path}\n\nPlease load a video file (.mp4, .mov, etc.) to convert it to frames.",
                    ).format(video_path=video_path),
                )
                return

        if not os.path.isfile(video_path):
            QMessageBox.warning(
                app,
                translate("VideoToFramesController", "Video to Frames"),
                translate(
                    "VideoToFramesController",
                    "The video file path is not valid:\n{video_path}\n\nPlease load a valid video file first.",
                ).format(video_path=video_path),
            )
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{base_name}_frames")

        reply = QMessageBox.question(
            app,
            translate("VideoToFramesController", "Convert Video to Frame Series"),
            translate(
                "VideoToFramesController",
                "This will convert the current video to a frame series:\n\nVideo: {video_name}\nOutput: {output_dir}\n\nThis may take several minutes and will overwrite any existing frame series.\n\nContinue?",
            ).format(video_name=os.path.basename(video_path), output_dir=output_dir),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        app.control_panel.video_to_frames_button.setEnabled(False)
        app.control_panel.video_to_frames_button.setText(translate("VideoToFramesController", "Converting..."))

        self.worker = VideoToFramesWorker(video_path, output_dir, quality=90)
        app.video_to_frames_worker = self.worker
        self.worker.progress_updated.connect(self.on_progress)
        self.worker.conversion_finished.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, message: str):
        """Handle progress updates from video conversion."""
        logging.info("Video conversion progress: %s", message)

    def on_finished(self, success: bool, message: str):
        """Handle completion of video conversion."""
        app = self.app
        app.control_panel.video_to_frames_button.setEnabled(True)
        app.control_panel.video_to_frames_button.setText(
            translate("VideoToFramesController", "Reset Video -> Frame Series")
        )

        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        app.video_to_frames_worker = None

        if success:
            QMessageBox.information(
                app,
                translate("VideoToFramesController", "Conversion Complete"),
                translate(
                    "VideoToFramesController",
                    "Video conversion completed successfully!\n\n{message}\n\nYou can now load the frame series for faster playback.",
                ).format(message=message),
            )
            logging.info("Video to frames conversion completed: %s", message)
        else:
            QMessageBox.critical(
                app,
                translate("VideoToFramesController", "Conversion Failed"),
                translate("VideoToFramesController", "Video conversion failed:\n\n{message}").format(message=message),
            )
            logging.error("Video to frames conversion failed: %s", message)
