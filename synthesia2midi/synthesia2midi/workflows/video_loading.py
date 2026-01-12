"""
Video loading workflow.

Handles video file loading, configuration management, and initial setup.
"""
import logging
import os
import sys
import subprocess
import hashlib
import traceback
import re
import threading
import time
from typing import Optional, Tuple

from PySide6.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PySide6.QtCore import Qt, QTimer

from synthesia2midi.video_loader import VideoSession, create_video_session
from synthesia2midi.core.app_state import AppState
from synthesia2midi.config_manager import ConfigManager


class VideoLoadingWorkflow:
    """
    Handles video loading and related configuration management.
    """
    
    def __init__(self, app_state: AppState, config_manager: ConfigManager, parent_widget=None):
        self.app_state = app_state
        self.config_manager = config_manager
        self.parent_widget = parent_widget
        self.logger = logging.getLogger(__name__)
        # Force INFO level for this logger to ensure our logs show
        self.logger.setLevel(logging.INFO)
    
    def load_video_file(self, filepath: str) -> Tuple[bool, Optional[VideoSession]]:
        """
        Load a video file and associated configuration.
        Automatically converts video files to frame sequences for better performance.
        
        Args:
            filepath: Path to video file or frame directory to load
            
        Returns:
            (success, video_session) tuple
        """
        self.logger.info(f"[VIDEO-LOAD-START] load_video_file called with: {filepath}")
        try:
            self.logger.info(f"[VIDEO-LOAD] Incoming path: {filepath}")
            self.logger.info(f"[VIDEO-LOAD] Path stats - exists: {os.path.exists(filepath)}, isfile: {os.path.isfile(filepath)}, isdir: {os.path.isdir(filepath)}")
            # Handle Windows WSL paths
            # When accessing WSL files from Windows, paths come in as \\wsl.localhost\...
            if filepath.startswith('\\\\wsl.localhost\\') or filepath.startswith('\\\\wsl$\\'):
                # Convert Windows WSL path to Linux path
                # Example: \\wsl.localhost\Ubuntu\home\<user>\... -> /home/<user>/...
                parts = filepath.replace('\\', '/').split('/')
                # Find where the Linux path starts (usually after the distro name)
                for i, part in enumerate(parts):
                    if part in ['Ubuntu', 'Debian', 'openSUSE', 'kali-linux']:
                        # Skip empty parts and distro name, then join the rest
                        filepath = '/' + '/'.join(parts[i+1:])
                        self.logger.info(f"[VIDEO-LOAD] Converted WSL path to Linux: {filepath}")
                        break
            
            # Validate file/directory exists
            if not os.path.exists(filepath):
                self._show_error("File Error", f"File not found: {filepath}")
                return False, None

            detected_fps = None
            
            # Track original video path for config loading
            original_video_path = None
            
            # Check if this is a video file that needs conversion
            if os.path.isfile(filepath) and filepath.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                original_video_path = filepath  # Save original path for config
                detected_fps = self._detect_video_fps(filepath)
                
                # Check if auto-conversion is enabled (default: True for better performance)
                auto_convert = getattr(self.app_state.ui, 'auto_convert_to_frames', True)
                self.logger.info(f"[VIDEO-LOAD] Auto-convert to frames enabled: {auto_convert}")
                self.logger.info(f"[VIDEO-LOAD] Video file path: {filepath}")
                self.logger.info(f"[VIDEO-LOAD] Video file exists: {os.path.exists(filepath)}")
                
                if auto_convert:
                    # Convert to FULL frames for navigation (optimization happens on-demand during detection)
                    self.logger.info("[VIDEO-LOAD] Starting frame conversion (full frames for navigation)...")
                    frames_path = self._convert_video_to_frames(filepath)
                    self.logger.info(f"[VIDEO-LOAD] Frame conversion returned: {frames_path}")
                    if frames_path:
                        filepath = frames_path  # Use frames instead of video
                        self.logger.info(f"[VIDEO-LOAD] Using frames directory: {frames_path}")
                        self._show_info("Performance Optimization",
                                      "Video has been converted to image sequence for better navigation.\n"
                                      "Detection will use on-demand cropping for additional speed.")
                    else:
                        self.logger.warning(f"[VIDEO-LOAD] Frame conversion failed, using original video")
                else:
                    self.logger.info("[VIDEO-LOAD] Auto-convert disabled; will continue with video file unless user chooses existing frames.")
                    # Check if frames already exist and offer to use them
                    base_name = os.path.splitext(os.path.basename(filepath))[0]
                    frames_dir = os.path.join(os.path.dirname(filepath), f"{base_name}_frames")
                    if os.path.exists(frames_dir):
                        from PySide6.QtWidgets import QMessageBox
                        reply = QMessageBox.question(self.parent_widget, "Use Existing Frames?",
                                                   f"Found existing frame sequence for this video.\n"
                                                   f"Use frames for better performance?",
                                                   QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes:
                            filepath = frames_dir
                            if detected_fps is None:
                                detected_fps = self._detect_video_fps(original_video_path)
                    
            # Create video session with full frames (no pre-cropping)
            self.logger.info(f"[VIDEO-LOAD] Creating video session for: {filepath}")
            video_session = create_video_session(filepath, fps=detected_fps)
            
            # Update app state with video information
            self._update_video_state(filepath, video_session, original_video_path)
            
            # Try to load associated configuration
            # Use original video path for config if available
            config_path = original_video_path if original_video_path else filepath
            config_loaded = self._load_associated_config(config_path)
            
            self.logger.info(f"Video loaded successfully: {filepath}")
            self.logger.info(f"Video info: {video_session.total_frames} frames, {video_session.fps} fps")
            
            if config_loaded:
                self.logger.info("Associated configuration loaded")
            else:
                self.logger.info("No associated configuration found")
            
            return True, video_session
            
        except Exception as e:
            self.logger.error(f"Failed to load video {filepath}: {e}")
            self._show_error("Video Loading Error", f"Failed to load video: {e}")
            return False, None
    
    def _update_video_state(self, filepath: str, video_session: VideoSession, original_video_path: Optional[str] = None):
        """Update app state with video file information."""
        # Clear per-video state before setting new paths to avoid carrying over old config.
        self.app_state.video.filepath_ini_used = None
        self.app_state.video.original_video_path = None

        # Store the actual path being used (frames or video)
        self.app_state.video.filepath = filepath
        
        # Store original video path if frames are being used (for config/MIDI association)
        if original_video_path:
            self.app_state.video.original_video_path = original_video_path
        else:
            # Check if this is a frames directory that came from a video
            if os.path.isdir(filepath) and filepath.endswith('_frames'):
                # Try to find the original video
                base_name = os.path.basename(filepath)[:-7]  # Remove '_frames'
                video_dir = os.path.dirname(filepath)
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    video_path = os.path.join(video_dir, base_name + ext)
                    if os.path.exists(video_path):
                        self.app_state.video.original_video_path = video_path
                        break
        
        self.app_state.video.current_frame_index = 0
        
        # Synchronize video properties from video_session to app_state
        self.app_state.video.fps = video_session.fps
        self.app_state.video.total_frames = video_session.total_frames
        
        # Clear any existing FPS override when loading a new video
        self.app_state.video.fps_override = None
        
        # Reset UI state for new video
        self.app_state.ui.selected_overlay_id = None
        self.app_state.calibration_mode = None
        self.app_state.current_calibration_key_type = None
        
        # Reset unsaved changes flag for new video
        self.app_state.unsaved_changes = False

    def _detect_video_fps(self, video_path: str) -> Optional[float]:
        """Detect FPS from a video file for propagation to image sequences."""
        try:
            session = VideoSession(video_path)
            fps = session.fps
            session.release()
            return fps
        except Exception as exc:
            self.logger.warning(f"[VIDEO-LOAD] Failed to detect FPS for {video_path}: {exc}")
            return None
    
    def _load_associated_config(self, video_filepath: str) -> bool:
        """
        Try to load configuration file associated with the video.
        
        Args:
            video_filepath: Path to the video file
            
        Returns:
            True if config was loaded, False otherwise
        """
        try:
            # Look for .ini file with same name as video
            base_name = os.path.splitext(video_filepath)[0]
            ini_path = base_name + ".ini"
            
            self.logger.info(f"[CONFIG-LOAD] Looking for configuration file:")
            self.logger.info(f"[CONFIG-LOAD]   Video filepath: {video_filepath}")
            self.logger.info(f"[CONFIG-LOAD]   Expected INI path: {ini_path}")
            self.logger.info(f"[CONFIG-LOAD]   INI exists: {os.path.exists(ini_path)}")
            
            if os.path.exists(ini_path):
                success = self.config_manager.load_config(ini_path, is_template=False)
                if success:
                    self.app_state.video.filepath_ini_used = ini_path
                    self.logger.info(f"Loaded configuration from {ini_path}")
                    return True
                else:
                    self.logger.warning(f"Failed to load configuration from {ini_path}")
            
            # Try other common configuration file patterns
            config_patterns = [
                base_name + "_config.ini",
                base_name + ".config",
                os.path.join(os.path.dirname(video_filepath), "config.ini")
            ]
            
            for config_path in config_patterns:
                if os.path.exists(config_path):
                    success = self.config_manager.load_config(config_path, is_template=False)
                    if success:
                        self.app_state.video.filepath_ini_used = config_path
                        self.logger.info(f"Loaded configuration from {config_path}")
                        return True
            
            self.logger.info(f"[CONFIG-LOAD] No configuration file found for {video_filepath}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading associated config: {e}")
            return False
    
    def save_current_config(self, output_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            output_path: Optional specific path to save to
            
        Returns:
            True if saved successfully, False otherwise
        """
        self.logger.debug(f"[CONFIG-SAVE] save_current_config called")
        self.logger.debug(f"[CONFIG-SAVE] Current hand detection state:")
        self.logger.debug(f"[CONFIG-SAVE]   left_hand_hue_mean: {self.app_state.detection.left_hand_hue_mean}")
        self.logger.debug(f"[CONFIG-SAVE]   right_hand_hue_mean: {self.app_state.detection.right_hand_hue_mean}")
        self.logger.debug(f"[CONFIG-SAVE]   hand_detection_calibrated: {self.app_state.detection.hand_detection_calibrated}")
        
        try:
            if output_path is None:
                # Generate default path based on video file
                if not self.app_state.video.filepath:
                    self._show_error("Save Error", "No video file loaded")
                    return False
                    
                # Use original video path for config if available (when using frames)
                if hasattr(self.app_state.video, 'original_video_path') and self.app_state.video.original_video_path:
                    # When using frame sequences, save config next to original video
                    video_path_for_config = self.app_state.video.original_video_path
                    base_name = os.path.splitext(self.app_state.video.original_video_path)[0]
                else:
                    # When using video directly
                    video_path_for_config = self.app_state.video.filepath
                    base_name = os.path.splitext(self.app_state.video.filepath)[0]
                output_path = base_name + ".ini"
            
            self.logger.info(f"[CONFIG-SAVE] Saving configuration:")
            self.logger.info(f"[CONFIG-SAVE]   Video path for config: {video_path_for_config}")
            self.logger.info(f"[CONFIG-SAVE]   Expected INI path: {output_path}")
            self.logger.debug(f"[CONFIG-SAVE] Calling config_manager.save_config with video path: {video_path_for_config}")
            success = self.config_manager.save_config(video_path_for_config)
            
            if success:
                self.app_state.video.filepath_ini_used = output_path
                self.app_state.unsaved_changes = False
                self.logger.info(f"Configuration saved to {output_path}")
                self.logger.info(f"[CONFIG-SAVE] [OK] Configuration saved successfully to {output_path}")
                return True
            else:
                self._show_error("Save Error", f"Failed to save configuration to {output_path}")
                self.logger.error(f"[CONFIG-SAVE] [FAILED] Failed to save configuration")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
            self._show_error("Save Error", f"Error saving configuration: {e}")
            return False
    
    def get_video_info(self) -> dict:
        """Get information about the currently loaded video."""
        if not self.app_state.video.filepath:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "filepath": self.app_state.video.filepath,
            "current_frame": self.app_state.video.current_frame_index,
            "config_file": self.app_state.video.filepath_ini_used,
            "overlays_count": len(self.app_state.overlays),
            "unsaved_changes": self.app_state.unsaved_changes
        }
    
    def close_video(self):
        """Close the current video and reset state."""
        self.app_state.video.filepath = ""
        self.app_state.video.filepath_ini_used = None
        self.app_state.video.current_frame_index = 0
        self.app_state.ui.selected_overlay_id = None
        self.app_state.calibration_mode = None
        self.app_state.current_calibration_key_type = None
        
        # Optionally clear overlays (could be preserved for next video)
        # self.app_state.overlays.clear()
        
        self.app_state.unsaved_changes = False
        self.logger.info("Video closed and state reset")
    
    def _convert_video_to_frames(self, video_path: str) -> Optional[str]:
        """
        Convert video to image sequence with improved performance.
        
        Key improvements:
        - Hardware acceleration detection
        - Multi-threaded encoding
        - FFmpeg progress parsing instead of file counting
        - Simplified progress monitoring
        """
        self.logger.warning(f"[FRAME-EXTRACT-START] === Starting video frame extraction ===")
        self.logger.warning(f"[FRAME-EXTRACT-START] Input video: {video_path}")
        
        # Handle Windows WSL paths
        if video_path.startswith('\\\\wsl.localhost\\') or video_path.startswith('\\\\wsl$\\'):
            # Convert Windows WSL path to Linux path
            parts = video_path.replace('\\', '/').split('/')
            for i, part in enumerate(parts):
                if part in ['Ubuntu', 'Debian', 'openSUSE', 'kali-linux']:
                    video_path = '/' + '/'.join(parts[i+1:])
                    self.logger.info(f"[FRAME-CONVERT] Converted WSL path to Linux: {video_path}")
                    break
        
        try:
            # Generate frames directory path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            video_dir = os.path.dirname(video_path)
            frames_dir = os.path.join(video_dir, f"{base_name}_frames")
            
            self.logger.info(f"[FRAME-CONVERT] Video path: {video_path}")
            self.logger.info(f"[FRAME-CONVERT] Base name: {base_name}")
            self.logger.info(f"[FRAME-CONVERT] Video directory: {video_dir}")
            self.logger.info(f"[FRAME-CONVERT] Frames directory will be: {frames_dir}")
            
            # Check if frames already exist
            if os.path.exists(frames_dir):
                # Verify it contains frames
                import glob
                frame_files = glob.glob(os.path.join(frames_dir, "frame_*.jpg")) + \
                             glob.glob(os.path.join(frames_dir, "frame_*.png"))
                if frame_files:
                    self.logger.info(f"Using existing frame sequence: {frames_dir}")
                    return frames_dir
                    
            # Check if ffmpeg is available
            from synthesia2midi.utils.ffmpeg_helper import check_ffmpeg_available
            is_available, message = check_ffmpeg_available()
            if not is_available:
                self.logger.warning(f"FFmpeg not available: {message}")
                self.logger.warning("Using video file directly")
                return None
                
            # Show progress dialog
            progress = QProgressDialog("Converting video to image sequence...\nThis is a one-time process.", 
                                     "Cancel", 0, 100, self.parent_widget)
            progress.setWindowTitle("Optimizing Performance")
            progress.setWindowModality(Qt.WindowModal)
            progress.setAutoClose(False)  # Disable auto-close to handle manually
            progress.setAutoReset(False)  # Prevent auto-reset at 100%
            progress.show()
            
            # Create frames directory
            self.logger.info(f"[FRAME-CONVERT] Creating frames directory: {frames_dir}")
            os.makedirs(frames_dir, exist_ok=True)
            
            # Verify directory was created
            if os.path.exists(frames_dir):
                self.logger.info(f"[FRAME-CONVERT] Frames directory created successfully")
            else:
                self.logger.error(f"[FRAME-CONVERT] Failed to create frames directory!")
                raise Exception(f"Failed to create directory: {frames_dir}")
            
            # Build ffmpeg command using our helper
            from synthesia2midi.utils.ffmpeg_helper import find_ffmpeg
            ffmpeg_path = find_ffmpeg()
            if not ffmpeg_path:
                raise FileNotFoundError("FFmpeg executable not found")
            
            self.logger.info(f"[FRAME-CONVERT] Using ffmpeg executable: {ffmpeg_path}")
            self.logger.info(f"[FRAME-CONVERT] FFmpeg exists: {os.path.exists(ffmpeg_path) if ffmpeg_path else False}")
                
            output_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
            self.logger.info(f"[FRAME-CONVERT] FFmpeg output pattern: {output_pattern}")
            
            # Check for hardware acceleration
            hwaccel_args = self._detect_hardware_acceleration(ffmpeg_path)
            
            # Try with hardware acceleration first, then fallback if it fails
            attempt_hwaccel = hwaccel_args is not None
            max_attempts = 2 if attempt_hwaccel else 1
            
            conversion_successful = False
            for attempt in range(max_attempts):
                # Build full command with detected ffmpeg path
                cmd = [ffmpeg_path]
                
                # Add hardware acceleration on first attempt if available
                if attempt == 0 and hwaccel_args:
                    cmd.extend(hwaccel_args)
                    self.logger.info(f"[FRAME-CONVERT] Attempt {attempt+1}: Using hardware acceleration: {' '.join(hwaccel_args)}")
                else:
                    if attempt > 0:
                        self.logger.info(f"[FRAME-CONVERT] Attempt {attempt+1}: Falling back to CPU-only processing")
                
                # Add input and encoding options
                cmd.extend([
                    '-i', video_path,
                    '-threads', '0',  # Use all CPU cores
                    '-q:v', '2',  # High quality JPEG
                    '-vf', 'format=bgr24',  # Ensure BGR format for OpenCV
                    '-progress', 'pipe:1',  # Output progress to stdout
                    '-nostats',  # Disable encoding stats
                    output_pattern
                ])
                
                # Run conversion in subprocess
                self.logger.info(f"Converting video to frames: {video_path} -> {frames_dir}")
                self.logger.info(f"[FRAME-CONVERT] Full ffmpeg command: {' '.join(cmd)}")
                
                # Log environment info (only on first attempt)
                if attempt == 0:
                    self.logger.info(f"[FRAME-CONVERT] Current working directory: {os.getcwd()}")
                    self.logger.info(f"[FRAME-CONVERT] Video file exists: {os.path.exists(video_path)}")
                    self.logger.info(f"[FRAME-CONVERT] Video file size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'}")
                    self.logger.info(f"[FRAME-CONVERT] Frames dir exists before ffmpeg: {os.path.exists(frames_dir)}")
                
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                         universal_newlines=True)
                
                # Quick check if process started successfully
                time.sleep(0.5)  # Give FFmpeg a moment to start
                poll_result = process.poll()
                if poll_result is not None and attempt == 0 and hwaccel_args:
                    # Process already exited, likely hardware acceleration failure
                    self.logger.warning(f"[FRAME-CONVERT] Hardware acceleration failed immediately (return code: {poll_result})")
                    
                    # Try to get early error output
                    try:
                        early_stderr = process.stderr.read(1000)  # Read first 1000 chars
                        if early_stderr:
                            self.logger.warning(f"[FRAME-CONVERT] Early error: {early_stderr[:200]}")
                    except:
                        pass
                    
                    # Clean up any partial output
                    import shutil
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    # Continue to next attempt without hardware acceleration
                    continue
                elif poll_result is not None:
                    # Process exited on non-hardware attempt
                    self.logger.error(f"[FRAME-CONVERT] FFmpeg failed to start on attempt {attempt+1} (return code: {poll_result})")
                    raise RuntimeError(f"FFmpeg failed to start with return code {poll_result}")
                
                # If we get here, process started successfully
                self.logger.info(f"[FRAME-CONVERT] FFmpeg process started successfully on attempt {attempt+1}")
                
                self.logger.debug(f"[PROGRESS] Starting conversion: video={video_path}")
                self.logger.debug(f"[PROGRESS] Output directory: {frames_dir}")
                
                # Variables for progress tracking
                duration_ms = None
                total_frames = None
                frames_created = 0
                fps = None
                stderr_output = []  # Store all stderr for error reporting
                
                # Parse FFmpeg stderr for video info (runs once at start)
                def parse_video_info():
                    nonlocal duration_ms, total_frames, fps, stderr_output
                    try:
                        # Read initial stderr for video information
                        while True:
                            line = process.stderr.readline()
                            if not line:
                                break
                            stderr_output.append(line)  # Store for later error reporting
                        
                            # Extract duration
                            if 'Duration:' in line and not duration_ms:
                                match = re.search(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)', line)
                                if match:
                                    hours, minutes, seconds = match.groups()
                                    duration_ms = (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000
                                    self.logger.info(f"[PROGRESS] Video duration: {duration_ms/1000:.1f} seconds")
                            
                            # Extract FPS and resolution
                            if 'Stream' in line and 'Video:' in line and not fps:
                                fps_match = re.search(r'(\d+(?:\.\d+)?)\s*fps', line)
                                if fps_match:
                                    fps = float(fps_match.group(1))
                                    if duration_ms:
                                        total_frames = int((duration_ms / 1000.0) * fps)
                                        self.logger.info(f"[PROGRESS] Estimated total frames: {total_frames} (at {fps} fps)")
                                        break
                    except Exception as e:
                        self.logger.error(f"[PROGRESS] Error parsing video info: {e}")
                
                # Start thread to parse video info
                info_thread = threading.Thread(target=parse_video_info)
                info_thread.daemon = True
                info_thread.start()
                
                # Parse FFmpeg progress output
                def parse_progress_output():
                    nonlocal frames_created
                    try:
                        for line in process.stdout:
                            # FFmpeg progress format: "frame=  123"
                            if line.startswith('frame='):
                                match = re.search(r'frame=\s*(\d+)', line)
                                if match:
                                    frames_created = int(match.group(1))
                    except Exception as e:
                        self.logger.error(f"[PROGRESS] Error parsing progress: {e}")
                
                # Start thread to parse progress
                progress_thread = threading.Thread(target=parse_progress_output)
                progress_thread.daemon = True
                progress_thread.start()
                
                # Wait a moment for video info to be parsed
                info_thread.join(timeout=2)
                
                # Main progress loop
                start_time = time.time()
                last_log_time = start_time
                
                while process.poll() is None:  # While FFmpeg is running
                    # Update progress dialog
                    if total_frames and total_frames > 0:
                        percent = min(99, int(frames_created * 100 / total_frames))
                        progress.setValue(percent)
                        progress.setLabelText(f"Converting video... {frames_created}/{total_frames} frames ({percent}%)")
                    else:
                        progress.setLabelText(f"Converting video... {frames_created} frames")
                
                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time > 5:
                        elapsed = current_time - start_time
                        conversion_fps = frames_created / elapsed if elapsed > 0 else 0
                        self.logger.info(f"[PROGRESS] {frames_created} frames converted at {conversion_fps:.1f} fps")
                        last_log_time = current_time
                    
                    # Check for user cancellation
                    if progress.wasCanceled():
                        self.logger.info("User canceled video conversion")
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        
                        # Clean up
                        import shutil
                        shutil.rmtree(frames_dir, ignore_errors=True)
                        progress.close()
                        return None
                    
                    # Process events and sleep
                    QApplication.processEvents()
                    time.sleep(0.1)
                
                # Process finished
                return_code = process.poll()
                self.logger.info(f"[FRAME-CONVERT] FFmpeg finished with return code: {return_code}")
                
                # Final frame count verification
                import glob
                final_frames = glob.glob(os.path.join(frames_dir, "frame_*.jpg"))
                final_count = len(final_frames)
                
                # Check if this was a hardware acceleration failure on first attempt
                if return_code != 0 and attempt == 0 and hwaccel_args and max_attempts > 1:
                    self.logger.warning(f"[FRAME-CONVERT] Hardware acceleration failed during processing, will retry with CPU")
                    progress.close()  # Close progress dialog before retry
                    
                    # Clean up failed attempt
                    import shutil
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    # Continue to next attempt
                    continue
                
                # Check for success
                if return_code == 0 and final_count > 0:
                    # Success
                    progress.setValue(100)
                    progress.close()

                    elapsed = time.time() - start_time
                    self.logger.info(f"[FRAME-CONVERT] Successfully converted {final_count} frames in {elapsed:.1f} seconds")
                    self.logger.info(f"[FRAME-CONVERT] Average conversion speed: {final_count/elapsed:.1f} fps")

                    conversion_successful = True
                    break  # Exit retry loop
                
                # If we're here and it's the last attempt, handle failure
                if attempt == max_attempts - 1:
                    # Failure on final attempt
                    progress.close()
                    
                    # Log error from stored stderr if available
                    if stderr_output:
                        stderr_text = ''.join(stderr_output[-20:])  # Last 20 lines
                        self.logger.error(f"[FRAME-CONVERT] FFmpeg error output:\n{stderr_text}")
                    
                    # Log specific failure reason
                    if return_code != 0:
                        self.logger.error(f"[FRAME-CONVERT] FFmpeg failed with return code: {return_code}")
                    if final_count == 0:
                        self.logger.error(f"[FRAME-CONVERT] No frames were created")
                    
                    # Clean up failed conversion
                    import shutil
                    shutil.rmtree(frames_dir, ignore_errors=True)
                    
                    if self.parent_widget:
                        QMessageBox.warning(self.parent_widget,
                                          "Frame Conversion Failed",
                                          f"Failed to convert video to frames.\n"
                                          f"Return code: {return_code}\n"
                                          f"Frames created: {final_count}\n\n"
                                          "The video will be loaded directly instead.")
                
            # End of attempt loop
            if conversion_successful:
                return frames_dir
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Error converting video to frames: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Clean up progress dialog if it exists
            try:
                if 'progress' in locals():
                    progress.close()
            except:
                pass
            
            # Show error to user
            if self.parent_widget:
                QMessageBox.warning(self.parent_widget,
                                  "Frame Conversion Failed",
                                  f"Failed to convert video to frames:\n{str(e)}\n\n"
                                  "The video will be loaded directly instead.")
            
            return None
    
    def _show_error(self, title: str, message: str):
        """Show error message (if parent widget available)."""
        if self.parent_widget:
            QMessageBox.critical(self.parent_widget, title, message)
        else:
            self.logger.error(f"{title}: {message}")
            
    def _show_info(self, title: str, message: str):
        """Show info message (if parent widget available)."""
        if self.parent_widget:
            QMessageBox.information(self.parent_widget, title, message)
        else:
            self.logger.info(f"{title}: {message}")
    
    def _detect_hardware_acceleration(self, ffmpeg_path: str) -> Optional[list]:
        """
        Detect available hardware acceleration options.
        
        Returns list of FFmpeg arguments for hardware acceleration, or None if unavailable.
        """
        try:
            # Check for NVIDIA GPU acceleration
            result = subprocess.run([ffmpeg_path, '-hwaccels'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                available_hwaccels = result.stdout.lower()
                
                # NVIDIA CUDA/NVENC
                if 'cuda' in available_hwaccels or 'nvenc' in available_hwaccels:
                    self.logger.info("[HWACCEL] NVIDIA hardware acceleration available")
                    return ['-hwaccel', 'cuda']
                
                # Intel Quick Sync
                elif 'qsv' in available_hwaccels:
                    self.logger.info("[HWACCEL] Intel Quick Sync acceleration available")
                    return ['-hwaccel', 'qsv']
                
                # AMD
                elif 'amf' in available_hwaccels:
                    self.logger.info("[HWACCEL] AMD acceleration available")
                    return ['-hwaccel', 'amf']
                
                # macOS VideoToolbox
                elif 'videotoolbox' in available_hwaccels:
                    self.logger.info("[HWACCEL] VideoToolbox acceleration available")
                    return ['-hwaccel', 'videotoolbox']
                
                # Generic auto detection
                elif 'auto' in available_hwaccels:
                    self.logger.info("[HWACCEL] Using auto hardware acceleration")
                    return ['-hwaccel', 'auto']
        except Exception as e:
            self.logger.debug(f"[HWACCEL] Could not detect hardware acceleration: {e}")
        
        return None
