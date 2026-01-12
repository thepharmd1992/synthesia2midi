"""
MIDI conversion workflow.

Handles the complete conversion process including detection, frame processing,
and MIDI file generation.
"""
import logging
import os
import shutil
from typing import Dict, Tuple, Optional, Callable, List

import cv2
import numpy as np
from PySide6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from PySide6.QtCore import Qt

from synthesia2midi.midi_generator import MidiWriter
from synthesia2midi.detection.factory import DetectionFactory
from synthesia2midi.video_loader import VideoSession
from synthesia2midi.app_config import DEBUG_FRAMES_DIR
from synthesia2midi.core.app_state import AppState
from synthesia2midi.detection.roi_utils import extract_roi_bgr, get_average_color_from_roi, euclidean_distance


class ConversionWorkflow:
    """
    Handles the complete MIDI conversion process.
    """
    
    def __init__(self, app_state: AppState, video_session: VideoSession, 
                 parent_widget=None, detection_manager=None):
        self.app_state = app_state
        self.video_session = video_session
        self.parent_widget = parent_widget
        self.detection_manager = detection_manager
        self.logger = logging.getLogger(f"{__name__}.ConversionWorkflow")
    
    def convert_to_midi(self, output_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Convert video to MIDI file.
        
        Args:
            output_path: Where to save MIDI file
            progress_callback: Optional callback for progress updates (frame_idx, total_frames)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.warning(f"\n[CONVERSION] === MIDI CONVERSION STARTING ===")
            self.logger.warning(f"[CONVERSION] Output path: {output_path}")
            
            
            # Set detection mode to conversion (full detection with all features)
            original_navigation_mode = True
            if self.detection_manager:
                original_navigation_mode = self.detection_manager._is_navigation_mode
                self.detection_manager.set_navigation_mode(False)
                self.logger.info("[CONVERSION] Set detection mode to conversion (full detection)")
            
            # Validate prerequisites
            self.logger.info("[CONVERSION] Validating prerequisites...")
            errors = self._validate_prerequisites()
            if errors:
                self.logger.warning(f"[CONVERSION] Validation failed with {len(errors)} errors:")
                for error in errors:
                    self.logger.debug(f"  - {error}")
                self._show_error("Conversion Error", "\\n".join(errors))
                # Restore navigation mode before returning
                if self.detection_manager:
                    self.detection_manager.set_navigation_mode(original_navigation_mode)
                return False
            self.logger.info("[CONVERSION] Validation passed")
            
            # Log overlay configuration
            self.logger.info(f"[CONVERSION] Total overlays: {len(self.app_state.overlays)}")
            self.logger.info(f"Starting conversion with {len(self.app_state.overlays)} overlays:")
            for overlay in sorted(self.app_state.overlays, key=lambda o: o.key_id):
                self.logger.debug(f"Overlay key_id={overlay.key_id}, note={overlay.get_full_note_name()}, "
                      f"x={overlay.x:.1f}, y={overlay.y:.1f}, type={overlay.key_type}")
            
            # Calculate frame range
            total_frames = self._calculate_total_frames()
            self.logger.info(f"[CONVERSION] Frame range: {self.app_state.video.processing_start_frame} to {self.app_state.video.processing_end_frame}")
            self.logger.info(f"[CONVERSION] Total frames to process: {total_frames}")
            self.logger.info(f"Starting MIDI conversion process... Processing {total_frames} frames.")
            
            # Setup debug directory if needed
            self._setup_debug_directory()
            
            # Setup MIDI writer
            self.logger.info("[CONVERSION] Setting up MIDI writer...")
            midi_writer = self._setup_midi_writer()
            
            # Use original video session and overlays at full resolution
            self.logger.info("[CONVERSION] Using original video at full resolution")
            
            # Setup detection method
            self.logger.info("[CONVERSION] Setting up detection method...")
            detector = self._setup_detection_method()
            self.logger.info(f"[CONVERSION] Detection method created: {detector.get_name()}")
            
            # Process frames and generate MIDI
            self.logger.info("[CONVERSION] Starting frame processing...")
            success = self._process_frame_range(detector, midi_writer, total_frames, progress_callback)
            
            if success:
                # Save MIDI file
                self.logger.info("[CONVERSION] Frame processing completed successfully")
                self.logger.info(f"[CONVERSION] Saving MIDI file to: {output_path}")
                self.logger.warning("[SAVE-START] Beginning MIDI file save operation...")
                try:
                    midi_writer.save_file(output_path)
                    self.logger.warning("[SAVE-COMPLETE] MIDI file saved successfully")
                except Exception as save_error:
                    self.logger.error(f"[SAVE-ERROR] Failed to save MIDI file: {save_error}", exc_info=True)
                    raise
                    
                self.logger.info("[SETTINGS-LOG] Saving settings log...")
                self._save_midi_settings_log(output_path, total_frames, detector.get_name())
                self.logger.info("[SETTINGS-LOG] Settings log saved")
                
                self.logger.info(f"MIDI conversion completed successfully: {output_path}")
                self.logger.info(f"[CONVERSION] SUCCESS - MIDI file saved: {output_path}")
                # Restore navigation mode
                if self.detection_manager:
                    self.detection_manager.set_navigation_mode(original_navigation_mode)
                return True
            else:
                self.logger.warning("[CONVERSION] Frame processing failed or was cancelled")
                self.logger.warning("MIDI conversion was cancelled or failed")
                # Restore navigation mode
                if self.detection_manager:
                    self.detection_manager.set_navigation_mode(original_navigation_mode)
                return False
                
        except Exception as e:
            self.logger.info(f"[CONVERSION] EXCEPTION: {type(e).__name__}: {str(e)}")
            import traceback
            self.logger.info("[CONVERSION] Traceback:")
            traceback.print_exc()
            self.logger.error(f"MIDI conversion failed: {e}")
            self._show_error("Conversion Error", f"MIDI conversion failed: {e}")
            # Restore navigation mode
            if self.detection_manager:
                self.detection_manager.set_navigation_mode(original_navigation_mode)
            return False
    
    def _validate_prerequisites(self) -> List[str]:
        """Validate that conversion can proceed."""
        errors = []
        
        if not self.video_session:
            errors.append("Please open a video file first.")
        
        if not self.app_state.overlays:
            errors.append("No overlays defined. Please run calibration wizard first.")
        else:
            missing_unlit = [
                overlay.key_id
                for overlay in self.app_state.overlays
                if overlay.unlit_reference_color is None
            ]
            if missing_unlit:
                errors.append(
                    "Unlit key calibration missing for overlays: "
                    + ", ".join(str(key_id) for key_id in missing_unlit)
                    + ". Please calibrate unlit keys before conversion."
                )
            if self.app_state.detection.use_histogram_detection:
                missing_unlit_hist = [
                    overlay.key_id
                    for overlay in self.app_state.overlays
                    if overlay.unlit_hist is None
                ]
                if missing_unlit_hist:
                    errors.append(
                        "Unlit histogram calibration missing for overlays: "
                        + ", ".join(str(key_id) for key_id in missing_unlit_hist)
                        + ". Please calibrate unlit keys before conversion."
                    )
        
        # Validate exemplar colors are calibrated - standard mode only (4 exemplars)
        required_exemplars = ["LW", "LB", "RW", "RB"]
        missing_exemplars = []
        for exemplar in required_exemplars:
            if (exemplar not in self.app_state.detection.exemplar_lit_colors or 
                self.app_state.detection.exemplar_lit_colors[exemplar] is None):
                missing_exemplars.append(exemplar)
        
        if missing_exemplars:
            exemplar_names = {"LW": "Left White", "LB": "Left Black", 
                            "RW": "Right White", "RB": "Right Black"}
            missing_names = [exemplar_names[e] for e in missing_exemplars]
            errors.append(f"Missing exemplar colors: {', '.join(missing_names)}. "
                        f"Please calibrate all exemplar colors before conversion.")
        
        # Basic validation of critical settings
        if not 0.1 <= self.app_state.detection.detection_threshold <= 0.99:
            errors.append(f"Detection threshold {self.app_state.detection.detection_threshold} must be between 0.1 and 0.99")
        
        if self.app_state.midi.tempo <= 0:
            errors.append("Tempo must be greater than 0")
        
        return errors
    
    def _calculate_total_frames(self) -> int:
        """Calculate total frames to process based on processing range settings."""
        start_frame = self.app_state.video.processing_start_frame
        end_frame = self.app_state.video.processing_end_frame
        video_total = self.video_session.total_frames
        
        # Validate and apply frame range
        if end_frame <= 0 or end_frame >= video_total:
            end_frame = video_total - 1
        
        if start_frame < 0:
            start_frame = 0
        
        if start_frame >= end_frame:
            # Invalid range, use full video
            return video_total
        
        # Return the trimmed frame count
        trimmed_count = end_frame - start_frame + 1
        self.logger.info(f"Processing trimmed range: frames {start_frame}-{end_frame} ({trimmed_count} of {video_total} total)")
        return trimmed_count
    
    def _setup_debug_directory(self):
        """Prepare the debug frame output directory (currently disabled)."""
        if True:  # Debug frame writing is currently disabled.
            return
            
        if os.path.exists(DEBUG_FRAMES_DIR):
            try:
                shutil.rmtree(DEBUG_FRAMES_DIR)
            except PermissionError as e:
                self.logger.warning(f"Could not remove debug_frames directory: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error removing debug_frames directory: {e}")
        
        try:
            os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Could not create debug_frames directory: {e}")
    
    def _setup_midi_writer(self) -> MidiWriter:
        """Setup and configure MIDI writer."""
        midi_writer = MidiWriter(midi_file_format=1)
        track = 0
        time = 0
        
        midi_writer.set_track_name(track, time, f"{os.path.basename(self.app_state.video.filepath)} MIDI")
        midi_writer.set_tempo(track, time, self.app_state.midi.tempo)
        
        # Setup instruments for each channel
        for channel_1_based in set(self.app_state.midi.color_to_channel_map.values()):
            midi_writer.add_program_change(track, channel_1_based - 1, time, 0)
        
        return midi_writer
    
    def _setup_detection_method(self):
        """Setup detection method based on configuration."""
        try:
            detector = DetectionFactory.create_from_app_state(
                self.app_state, 
                self.app_state.overlays
            )
            self.logger.info(f"Created detection method: {detector.get_name()}")
            
            # Log detailed detection method info for MIDI conversion
            method_name = detector.__class__.__name__
            self.logger.info(f"[MIDI-CONVERSION] Using detection method: {method_name}")
            
            return detector
        except Exception as e:
            self.logger.error(f"Failed to create detection method: {e}")
            self.logger.info("Falling back to standard detection")
            return DetectionFactory.create_detector('standard')
    
    def _process_frame_range(self, detector, midi_writer: MidiWriter, total_frames: int, 
                           progress_callback: Optional[Callable]) -> bool:
        """Process the specified frame range and generate MIDI events."""
        self.logger.info(f"\n[FRAME-PROCESS] Starting frame processing with {detector.get_name()}")
        
        # Use original video session and overlays at full resolution
        video_session = self.video_session
        detection_overlays = self.app_state.overlays
        
        self.logger.warning("[FRAME-PROCESS] *** USING ORIGINAL VIDEO AT FULL RESOLUTION ***")
        # Handle both VideoSession (has filepath) and ImageSequenceSession (has image_pattern)
        video_path = getattr(self.video_session, 'filepath', None) or getattr(self.video_session, 'image_pattern', 'Unknown')
        self.logger.warning(f"[FRAME-PROCESS] Video path: {video_path}")
        self.logger.warning(f"[FRAME-PROCESS] Resolution: {self.video_session.width}x{self.video_session.height}")
        self.logger.warning(f"[FRAME-PROCESS] Using {len(detection_overlays)} overlays at full resolution")
        
        # Track note states: {key_id: (start_frame, midi_channel_0_based)}
        active_notes: Dict[int, Tuple[int, int]] = {}
        
        # Create progress dialog if no callback provided
        progress_dialog = None
        if progress_callback is None and self.parent_widget:
            progress_dialog = QProgressDialog("Detecting notes...", "Cancel", 0, total_frames, self.parent_widget)
            progress_dialog.setWindowTitle("Processing...")
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.show()
        
        # Get actual frame range for processing
        start_frame = self.app_state.video.processing_start_frame
        end_frame = self.app_state.video.processing_end_frame
        video_total = video_session.total_frames
        
        # Validate frame range
        if end_frame <= 0 or end_frame >= video_total:
            end_frame = video_total - 1
        if start_frame < 0:
            start_frame = 0
        if start_frame >= end_frame:
            start_frame = 0
            end_frame = video_total - 1
        
        # Log processing details for large videos
        self.logger.info(f"Starting frame processing: {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
        self.logger.info(f"Video total frames: {video_total}, Processing {total_frames} frames")
        
        # Seek to start frame for sequential reading (massive performance improvement)
        video_session.seek_to_frame(start_frame)
        
        # Add timing information for large videos
        import time
        processing_start_time = time.time()
        last_log_time = processing_start_time
        
        try:
            self.logger.debug(f"[FRAME-PROCESS] Processing frames {start_frame} to {end_frame} ({total_frames} total)")
            total_notes_created = 0
            frames_with_detections = 0
            
            for frame_count, actual_frame_idx in enumerate(range(start_frame, end_frame + 1)):
                # Check for cancellation
                if progress_dialog and progress_dialog.wasCanceled():
                    self.logger.debug("[FRAME-PROCESS] Cancelled by user")
                    self.logger.info("MIDI conversion cancelled by user.")
                    return False
                
                # Enhanced progress logging for large videos
                current_time = time.time()
                if frame_count % 100 == 0:  # Log every 100 frames
                    elapsed = current_time - processing_start_time
                    fps_rate = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_rate if fps_rate > 0 else 0
                    percent_complete = frame_count/total_frames*100
                    self.logger.info(f"Progress: {frame_count}/{total_frames} frames ({percent_complete:.1f}%), "
                                   f"Rate: {fps_rate:.1f} fps, ETA: {eta/60:.1f} minutes")
                    
                    # Special logging for near completion
                    if percent_complete >= 95.0:
                        self.logger.warning(f"[NEAR-COMPLETION] Processing at {percent_complete:.2f}% - Frame {frame_count} of {total_frames}")
                        self.logger.warning(f"[NEAR-COMPLETION] Actual frame index: {actual_frame_idx}, Active notes: {len(active_notes)}")
                
                # Update progress dialog more frequently for responsiveness
                if frame_count % 10 == 0:  # Changed from 20 to 10
                    if progress_callback:
                        progress_callback(frame_count, total_frames)
                    elif progress_dialog:
                        phase_label = "Detecting notes" if detector else "Processing"
                        percent = frame_count/total_frames*100
                        progress_dialog.setLabelText(f"{phase_label}... {frame_count}/{total_frames} ({percent:.1f}%) (frame {actual_frame_idx})")
                        progress_dialog.setValue(frame_count)
                        # Process events with a timeout to prevent UI freezing
                        QApplication.processEvents()
                        # Force immediate repaint for large videos
                        if frame_count % 100 == 0:
                            progress_dialog.repaint()
                        # Extra logging when stuck near 99%
                        if percent >= 99.0:
                            self.logger.warning(f"[PROGRESS-99%] Dialog update at {percent:.2f}% - frame {frame_count}/{total_frames}")
                
                # Read frame
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[FRAME-READ-99%] About to read frame {actual_frame_idx} at {frame_count/total_frames*100:.2f}%")
                
                # Time frame reading
                frame_read_start = time.time()
                success, frame_bgr = video_session.get_frame_sequential()
                frame_read_time = (time.time() - frame_read_start) * 1000  # Convert to ms
                
                # Log frame read timing for first few frames and periodically
                if frame_count < 5 or frame_count % 500 == 0:
                    self.logger.warning(f"[FRAME-READ-TIMING] Frame {frame_count} read from ORIGINAL in {frame_read_time:.2f}ms")
                
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[FRAME-READ-99%-COMPLETE] Frame read complete: success={success}, has_frame={frame_bgr is not None}")
                    
                if not success or frame_bgr is None:
                    self.logger.warning(f"Could not read frame {actual_frame_idx} (count {frame_count}). Success={success}, frame is None={frame_bgr is None}")
                    # For large videos, log if we're hitting read issues frequently
                    if frame_count > 0 and frame_count % 1000 == 0:
                        self.logger.error(f"Frame read failure at frame {actual_frame_idx}. This may indicate video corruption or memory issues.")
                    continue
                
                # Log first frame detection method
                if frame_count == 0:
                    detector_type = detector.__class__.__name__
                    self.logger.info(f"[FRAME-PROCESSING] Starting frame processing with {detector_type}")
                
                # Use original frame directly for detection
                detection_frame = frame_bgr
                # detection_overlays set to original overlays at full resolution
                
                # Detect pressed keys using unified interface with optimized frame
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[DETECTION-99%] Starting detection at {frame_count/total_frames*100:.2f}%")
                
                # TIMING: Start detection timing
                detection_start_time = time.time()
                pressed_key_ids = detector.detect_frame(
                    detection_frame,  # Use full resolution frame for detection
                    detection_overlays,  # Use original overlays at full resolution
                    self.app_state.detection.exemplar_lit_colors,
                    self.app_state.detection.exemplar_lit_histograms,
                    self.app_state.detection.detection_threshold,
                    hist_ratio_threshold=self.app_state.detection.hist_ratio_threshold,
                    rise_delta_threshold=self.app_state.detection.rise_delta_threshold,
                    fall_delta_threshold=self.app_state.detection.fall_delta_threshold,
                    use_histogram_detection=self.app_state.detection.use_histogram_detection,
                    use_delta_detection=self.app_state.detection.use_delta_detection,
                    similarity_ratio=self.app_state.detection.similarity_ratio,
                    apply_black_filter=self.app_state.detection.winner_takes_black_enabled,
                    # Pass hand detection parameters for exemplar selection
                    hand_assignment_enabled=self.app_state.detection.hand_assignment_enabled,
                    hand_detection_calibrated=self.app_state.detection.hand_detection_calibrated,
                    left_hand_hue_mean=self.app_state.detection.left_hand_hue_mean,
                    right_hand_hue_mean=self.app_state.detection.right_hand_hue_mean,
                )
                detection_end_time = time.time()
                detection_duration = (detection_end_time - detection_start_time) * 1000
                
                # Log timing information for first few frames and periodically
                if frame_count < 5 or frame_count % 100 == 0:
                    frame_info = f"full resolution {detection_frame.shape}"
                    self.logger.warning(f"[FRAME-TIMING] Frame {frame_count}: detection={detection_duration:.2f}ms, frame={frame_info}")
                
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[DETECTION-99%-COMPLETE] Detection complete, found {len(pressed_key_ids)} keys")
                
                # Count detections
                if pressed_key_ids:
                    frames_with_detections += 1
                
                # Log periodic progress
                if frame_count % 100 == 0:
                    self.logger.debug(f"[FRAME-PROCESS] Frame {actual_frame_idx}: {len(pressed_key_ids)} keys pressed, {len(active_notes)} notes active")
                
                # Generate MIDI events (use actual frame index for timing)
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[MIDI-EVENTS-99%] Processing MIDI events at {frame_count/total_frames*100:.2f}%")
                    
                notes_created_this_frame = self._process_midi_events(pressed_key_ids, actual_frame_idx, active_notes, midi_writer, frame_bgr, detection_overlays)
                total_notes_created += notes_created_this_frame
                
                if frame_count/total_frames*100 >= 99.0:
                    self.logger.warning(f"[MIDI-EVENTS-99%-COMPLETE] MIDI events processed, created {notes_created_this_frame} notes")
                
                # Save debug frame if enabled
                if False and actual_frame_idx % 100 == 0:  # Debug frame writing is currently disabled.
                    self._save_debug_frame(frame_bgr, actual_frame_idx, pressed_key_ids)
            
            # Log completion of main loop
            self.logger.warning(f"[FRAME-LOOP-COMPLETE] Finished processing all frames. Total processed: {frame_count}")
            self.logger.warning(f"[FRAME-LOOP-COMPLETE] Active notes before cleanup: {len(active_notes)}")
            
            # End any remaining active notes (use the last actual frame index)
            self.logger.info("[FINALIZATION] Starting to end remaining active notes...")
            self._end_remaining_notes(active_notes, end_frame, midi_writer)
            self.logger.info(f"[FINALIZATION] Ended {len(active_notes)} remaining notes")
            
            # Finalize any notes still tracked by the MIDI writer itself
            self.logger.info("[FINALIZATION] Finalizing MIDI writer notes...")
            effective_fps = self.app_state.video.fps_override if self.app_state.video.fps_override else self.video_session.fps
            final_time_beats = ((total_frames - 1) / effective_fps) * (self.app_state.midi.tempo / 60)
            midi_writer.finalize_active_notes(final_time_beats)
            self.logger.info("[FINALIZATION] MIDI writer finalization complete")
            
            # Log completion statistics
            total_elapsed = time.time() - processing_start_time
            avg_fps = total_frames / total_elapsed if total_elapsed > 0 else 0
            self.logger.info(f"Frame processing completed! Processed {total_frames} frames in {total_elapsed/60:.1f} minutes")
            self.logger.info(f"Average processing rate: {avg_fps:.1f} fps")
            
            print(f"\n[FRAME-PROCESS] Processing complete:")
            print(f"  - Total frames processed: {total_frames}")
            print(f"  - Frames with detections: {frames_with_detections}")
            print(f"  - Total notes created: {total_notes_created}")
            print(f"  - Processing time: {total_elapsed:.1f}s ({avg_fps:.1f} fps)")
            
            if total_notes_created == 0:
                print("[FRAME-PROCESS] WARNING: No notes were detected!")
                print("[FRAME-PROCESS] Check:")
                print("  - Are overlays placed correctly?")
                print("  - Is detection sensitivity appropriate?")
                print("  - Is the video range correct?")
            
            return True
            
        except Exception as e:
            self.logger.debug(f"[FRAME-PROCESS] EXCEPTION during frame processing: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error during frame processing: {e}", exc_info=True)
            return False
            
        finally:
            self.logger.warning("[FRAME-PROCESS-FINALLY] Entering finally block")
            if progress_dialog:
                self.logger.info("[FRAME-PROCESS-FINALLY] Closing progress dialog")
                progress_dialog.close()
            self.logger.warning("[FRAME-PROCESS-FINALLY] Exiting _process_frame_range method")
    
    
    def _process_midi_events(self, pressed_key_ids, frame_idx, active_notes, midi_writer, frame_bgr, overlays):
        """Process MIDI events for the current frame."""
        notes_created = 0
        
        # Use FPS override if set, otherwise use detected FPS
        effective_fps = self.app_state.video.fps_override if self.app_state.video.fps_override else self.video_session.fps
        
        # Log FPS values on first frame
        if frame_idx == self.app_state.video.processing_start_frame:
            self.logger.info(f"FPS Debug: detected={self.video_session.fps}, override={self.app_state.video.fps_override}, effective={effective_fps}")
        
        # Calculate time relative to the start of the trimmed range
        relative_frame_idx = frame_idx - self.app_state.video.processing_start_frame
        current_time_beats = (relative_frame_idx / effective_fps) * (self.app_state.midi.tempo / 60)
        
        # Check for note-offs (keys that were pressed but aren't now)
        keys_to_end = []
        for key_id in active_notes:
            if key_id not in pressed_key_ids:
                keys_to_end.append(key_id)
        
        for key_id in keys_to_end:
            start_frame, midi_channel = active_notes.pop(key_id)
            
            # Find the overlay to get MIDI note number
            overlay = next((o for o in overlays if o.key_id == key_id), None)
            if overlay:
                # Apply octave transpose when calculating MIDI note
                midi_note = overlay.get_midi_note_number(self.app_state.midi.octave_transpose)
                self.logger.debug(f"Frame {frame_idx}: Note OFF - key_id={key_id}, note={overlay.get_full_note_name(self.app_state.midi.octave_transpose)}, midi_note={midi_note}, channel={midi_channel}")
                midi_writer.add_note_off(0, midi_channel, current_time_beats, midi_note, 80)
        
        # Check for note-ons (newly pressed keys)
        for key_id in pressed_key_ids:
            if key_id not in active_notes:
                overlay = next((o for o in overlays if o.key_id == key_id), None)
                if overlay:
                    # Apply octave transpose when calculating MIDI note
                    midi_note = overlay.get_midi_note_number(self.app_state.midi.octave_transpose)
                    
                    # Log transpose debug info on first note
                    if not hasattr(self, '_logged_transpose_info'):
                        self.logger.info(f"[TRANSPOSE DEBUG] First note ON:")
                        self.logger.info(f"  - Overlay: key_id={key_id}, original={overlay.note_name_in_octave}{overlay.note_octave}")
                        self.logger.info(f"  - Transpose: {self.app_state.midi.octave_transpose} octaves")
                        self.logger.info(f"  - Display label: {overlay.get_full_note_name(self.app_state.midi.octave_transpose)}")
                        self.logger.info(f"  - MIDI note: {midi_note}")
                        self._logged_transpose_info = True
                    
                    # Determine MIDI channel based on color matching
                    midi_channel = self._determine_hand_channel(overlay, frame_bgr)
                    
                    self.logger.debug(f"Frame {frame_idx}: Note ON - key_id={key_id}, note={overlay.get_full_note_name(self.app_state.midi.octave_transpose)}, midi_note={midi_note}, channel={midi_channel}")
                    midi_writer.add_note_on(0, midi_channel, current_time_beats, midi_note, 80)
                    active_notes[key_id] = (frame_idx, midi_channel)
                    notes_created += 1
        
        return notes_created
    
    def _determine_hand_channel(self, overlay, frame_bgr) -> int:
        """
        Determine MIDI channel based on color matching to calibrated exemplars.
        Groups left/right hand colors together (LW+LB = channel 0, RW+RB = channel 1).
        
        Returns:
            Channel number (0-15) based on which color type the key matches
        """
        # If hand assignment is disabled, return default channel 0
        if not self.app_state.detection.hand_assignment_enabled:
            return 0
        
        # Extract current color from the overlay
        roi = extract_roi_bgr(frame_bgr, overlay)
        if roi is None:
            return 0  # Default channel if ROI extraction fails
        
        current_color = get_average_color_from_roi(frame_bgr, overlay)
        if current_color is None:
            return 0  # Default channel if color extraction fails
        
        # Find the closest exemplar color to the current color
        min_distance = float('inf')
        closest_exemplar = None
        
        # Debug logging for hand assignment
        self.logger.warning(f"[HAND-ASSIGNMENT] Key {overlay.key_id}: current_color={current_color}")
        self.logger.warning(f"[HAND-ASSIGNMENT] Available exemplars: {list(self.app_state.detection.exemplar_lit_colors.keys())}")
        
        for exemplar_type, exemplar_color in self.app_state.detection.exemplar_lit_colors.items():
            if exemplar_color is not None:
                distance = euclidean_distance(current_color, exemplar_color)
                self.logger.warning(f"[HAND-ASSIGNMENT] {exemplar_type}: color={exemplar_color}, distance={distance:.2f}")
                if distance < min_distance:
                    min_distance = distance
                    closest_exemplar = exemplar_type
        
        # Map exemplar type to channel based on color grouping
        if closest_exemplar:
            self.logger.warning(f"[HAND-ASSIGNMENT] Key {overlay.key_id}: closest_exemplar={closest_exemplar}, min_distance={min_distance:.2f}")
            # Left hand keys (both white and black) map to channel 0
            if closest_exemplar in ["LW", "LB"]:
                self.logger.warning(f"[HAND-ASSIGNMENT] Key {overlay.key_id}: assigned to LEFT HAND (channel 0)")
                return 0
            # Right hand keys (both white and black) map to channel 1
            elif closest_exemplar in ["RW", "RB"]:
                self.logger.warning(f"[HAND-ASSIGNMENT] Key {overlay.key_id}: assigned to RIGHT HAND (channel 1)")
                return 1
            # Additional colors map to channels 2, 3, 4, etc.
            elif closest_exemplar.startswith("COLOR_"):
                try:
                    # Extract color number from COLOR_N_W or COLOR_N_B format
                    parts = closest_exemplar.split("_")
                    if len(parts) >= 2:
                        color_num = int(parts[1])
                        # COLOR_3_W/B → channel 2, COLOR_4_W/B → channel 3, etc.
                        return color_num - 1
                except (ValueError, IndexError):
                    return 0
        
        # Default to channel 0 if no match found
        self.logger.warning(f"[HAND-ASSIGNMENT] Key {overlay.key_id}: NO MATCH FOUND, defaulting to channel 0")
        return 0
    
    def _end_remaining_notes(self, active_notes, final_frame, midi_writer):
        """End any notes that are still active at the end of processing."""
        effective_fps = self.app_state.video.fps_override if self.app_state.video.fps_override else self.video_session.fps
        # Calculate time relative to the start of the trimmed range
        relative_final_frame = final_frame - self.app_state.video.processing_start_frame
        final_time_beats = (relative_final_frame / effective_fps) * (self.app_state.midi.tempo / 60)
        
        for key_id, (start_frame, midi_channel) in active_notes.items():
            overlay = next((o for o in self.app_state.overlays if o.key_id == key_id), None)
            if overlay:
                # Apply octave transpose when calculating MIDI note
                midi_note = overlay.get_midi_note_number(self.app_state.midi.octave_transpose)
                midi_writer.add_note_off(0, midi_channel, final_time_beats, midi_note, 80)
    
    def _save_debug_frame(self, frame_bgr, frame_idx, pressed_key_ids):
        """Save debug frame if debug visuals are enabled."""
        # This would implement debug frame saving logic
        # For now, just log
        self.logger.debug(f"Debug frame {frame_idx}: {len(pressed_key_ids)} keys pressed")
    
    def _save_midi_settings_log(self, midi_path: str, frames_processed: int, detector_name: str):
        """Save a comprehensive log of MIDI settings used for this conversion in JSON format."""
        import json
        from datetime import datetime
        
        self.logger.info(f"[SETTINGS-LOG-START] Starting to save settings log for {midi_path}")
        log_path = midi_path.replace('.mid', '_settings.json')
        try:
            # Helper function to convert numpy arrays to lists for JSON serialization
            def serialize_for_json(obj):
                if hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                return obj
            
            # Create comprehensive settings data structure
            settings_data = {
                "metadata": {
                    "generated": datetime.now().isoformat(),
                    "midi_file": midi_path,
                    "video_file": self.app_state.video.filepath,
                    "frames_processed": frames_processed,
                    "detection_method": detector_name,
                    "total_overlays": len(self.app_state.overlays)
                },
                "detection_parameters": {
                    "detection_threshold": self.app_state.detection.detection_threshold,
                    "histogram_detection": {
                        "enabled": self.app_state.detection.use_histogram_detection,
                        "histogram_ratio_threshold": self.app_state.detection.hist_ratio_threshold
                    },
                    "delta_detection": {
                        "enabled": self.app_state.detection.use_delta_detection,
                        "rise_delta_threshold": self.app_state.detection.rise_delta_threshold,
                        "fall_delta_threshold": self.app_state.detection.fall_delta_threshold,
                    },
                    "black_key_filter": {
                        "enabled": self.app_state.detection.winner_takes_black_enabled,
                        "similarity_ratio": self.app_state.detection.similarity_ratio
                    },
                    "spark_detection": {
                        "enabled": self.app_state.detection.spark_detection_enabled,
                        "roi_top": self.app_state.detection.spark_roi_top,
                        "roi_bottom": self.app_state.detection.spark_roi_bottom,
                        "roi_visible": self.app_state.detection.spark_roi_visible,
                        "brightness_threshold": self.app_state.detection.spark_brightness_threshold,
                        "spark_detection_sensitivity": self.app_state.detection.spark_detection_sensitivity,
                        "detection_confidence": self.app_state.detection.spark_detection_confidence,
                        "calibration_data": {
                            "background": serialize_for_json(self.app_state.detection.spark_calibration_background),
                            "bar_only": serialize_for_json(self.app_state.detection.spark_calibration_bar_only),
                            "dimmest_sparks": serialize_for_json(self.app_state.detection.spark_calibration_dimmest_sparks),
                            "lh_bar_only": serialize_for_json(self.app_state.detection.spark_calibration_lh_bar_only),
                            "lh_dimmest_sparks": serialize_for_json(self.app_state.detection.spark_calibration_lh_dimmest_sparks),
                            "lh_brightest_sparks": serialize_for_json(self.app_state.detection.spark_calibration_lh_brightest_sparks),
                            "rh_bar_only": serialize_for_json(self.app_state.detection.spark_calibration_rh_bar_only),
                            "rh_dimmest_sparks": serialize_for_json(self.app_state.detection.spark_calibration_rh_dimmest_sparks),
                            "rh_brightest_sparks": serialize_for_json(self.app_state.detection.spark_calibration_rh_brightest_sparks)
                        }
                    },
                    "exemplar_lit_colors": {
                        "LW": serialize_for_json(self.app_state.detection.exemplar_lit_colors.get("LW")),
                        "LB": serialize_for_json(self.app_state.detection.exemplar_lit_colors.get("LB")),
                        "RW": serialize_for_json(self.app_state.detection.exemplar_lit_colors.get("RW")),
                        "RB": serialize_for_json(self.app_state.detection.exemplar_lit_colors.get("RB"))
                    }
                },
                "midi_settings": {
                    "tempo": self.app_state.midi.tempo,
                    "total_keys": self.app_state.midi.total_keys,
                    "leftmost_note_name": self.app_state.midi.leftmost_note_name,
                    "leftmost_note_octave": self.app_state.midi.leftmost_note_octave,
                    "color_to_channel_map": {str(k): v for k, v in self.app_state.midi.color_to_channel_map.items()}
                },
                "video_settings": {
                    "filepath": self.app_state.video.filepath,
                    "fps": self.app_state.video.fps,
                    "fps_override": self.app_state.video.fps_override,
                    "effective_fps": self.app_state.video.fps_override if self.app_state.video.fps_override else self.video_session.fps,
                    "total_frames": self.app_state.video.total_frames,
                    "current_frame_index": self.app_state.video.current_frame_index,
                    "frame_navigation_interval": self.app_state.video.current_nav_interval,
                    "processing_range": {
                        "start_frame": self.app_state.video.processing_start_frame,
                        "end_frame": self.app_state.video.processing_end_frame
                    },
                    "trim_settings": {
                        "start_frame": self.app_state.video.trim_start_frame,
                        "end_frame": self.app_state.video.trim_end_frame,
                        "video_is_trimmed": self.app_state.video.video_is_trimmed
                    }
                },
                "ui_settings": {
                    "show_overlays": self.app_state.ui.show_overlays,
                    "live_detection_feedback": self.app_state.ui.live_detection_feedback,
                    "visual_threshold_monitor_enabled": self.app_state.ui.visual_threshold_monitor_enabled,
                    "selected_overlay_id": self.app_state.ui.selected_overlay_id
                },
                "calibration_settings": {
                    "calibration_mode": self.app_state.calibration.calibration_mode,
                    "current_calibration_key_type": self.app_state.calibration.current_calibration_key_type,
                    "calib_start_frame": self.app_state.calibration.calib_start_frame,
                    "calib_end_frame": self.app_state.calibration.calib_end_frame
                },
                "testing_notes": {
                    "description": "Use this section to annotate frame ranges that deviated from ground truth",
                    "deviations": [
                        {
                            "frame_range": "",
                            "issue": "",
                            "notes": ""
                        },
                        {
                            "frame_range": "",
                            "issue": "",
                            "notes": ""
                        },
                        {
                            "frame_range": "",
                            "issue": "",
                            "notes": ""
                        },
                        {
                            "frame_range": "",
                            "issue": "",
                            "notes": ""
                        },
                        {
                            "frame_range": "",
                            "issue": "",
                            "notes": ""
                        }
                    ]
                }
            }
            
            self.logger.info("[SETTINGS-LOG] Writing JSON data to file...")
            with open(log_path, 'w') as f:
                json.dump(settings_data, f, indent=2)
                
            self.logger.info(f"[SETTINGS-LOG-SUCCESS] Comprehensive settings saved to: {log_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save settings log: {e}")
            # Fallback to simple text log if JSON fails
            try:
                log_path_fallback = midi_path.replace('.mid', '_settings.log')
                with open(log_path_fallback, 'w') as f:
                    f.write(f"MIDI Conversion Settings Log (Fallback)\\n")
                    f.write(f"Generated: {datetime.now()}\\n\\n")
                    f.write(f"Video: {self.app_state.video.filepath}\\n")
                    f.write(f"Detection threshold: {self.app_state.detection.detection_threshold}\\n")
                    f.write(f"Histogram detection: {self.app_state.detection.use_histogram_detection}\\n")
                    f.write(f"Error saving JSON: {e}\\n")
            except:
                pass
    
    
    def _show_error(self, title: str, message: str):
        """Show error message (if parent widget available)."""
        if self.parent_widget:
            QMessageBox.critical(self.parent_widget, title, message)
        else:
            self.logger.error(f"{title}: {message}")
