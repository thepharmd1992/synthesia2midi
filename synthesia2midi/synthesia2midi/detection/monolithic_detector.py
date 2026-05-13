#!/usr/bin/env python3
"""
Manual ROI piano keyboard auto-detector facade.

This module preserves the historical ``MonolithicPianoDetector`` public API while
composing focused detector stages from smaller modules:
- black-key detection
- white-key geometry/lattice/boundary solvers
- musical note assignment
- visualization output
"""

import logging

import cv2

from .black_key_detector import BlackKeyDetectionMixin
from .black_note_assignment import BlackNoteAssignmentMixin
from .black_note_center_map import BlackNoteCenterMapMixin
from .black_residual_warp import BlackResidualWarpMixin
from .detector_defaults import DEFAULT_DETECTION_PARAMS
from .detector_geometry import DetectorGeometryMixin
from .detector_visualization import DetectorVisualizationMixin
from .note_assignment import NoteAssignmentMixin
from .note_parsing import NoteParsingMixin
from .white_key_boundary_solver import WhiteKeyBoundarySolverMixin
from .white_key_geometry import WhiteKeyGeometryMixin
from .white_key_lattice_model import WhiteKeyLatticeModelMixin
from .white_key_lattice_solver import WhiteKeyLatticeSolverMixin
from .white_note_assignment import WhiteNoteAssignmentMixin


class MonolithicPianoDetector(
    DetectorGeometryMixin,
    BlackKeyDetectionMixin,
    WhiteKeyGeometryMixin,
    NoteParsingMixin,
    BlackNoteAssignmentMixin,
    BlackNoteCenterMapMixin,
    BlackResidualWarpMixin,
    WhiteKeyLatticeModelMixin,
    WhiteKeyLatticeSolverMixin,
    WhiteKeyBoundarySolverMixin,
    WhiteNoteAssignmentMixin,
    NoteAssignmentMixin,
    DetectorVisualizationMixin,
):
    """
    Compatibility facade for manual ROI piano-key detection.

    The class intentionally keeps legacy method names and state attributes so
    existing calibration wizard code, tests, and tuning profiles continue to work
    while implementation responsibilities live in focused modules.

    Args:
        image_path: Path to the image file to analyze.
        keyboard_region: Tuple of (top_y, bottom_y, left_x, right_x) defining
            the manual ROI for keyboard detection.
        detection_profile: Optional detector-parameter overrides.
    """

    def __init__(self, image_path, keyboard_region=None, detection_profile=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.height, self.width = self.gray.shape
        self.logger.debug(f"Analyzing image: {self.width}x{self.height} pixels")

        # Detection results
        self.black_keys = []
        self.white_keys = []
        self.keyboard_region = keyboard_region  # Must be provided for manual ROI
        self.key_notes = {}
        # Detection parameters (allow overrides for low-quality fallbacks)
        self.params = {**DEFAULT_DETECTION_PARAMS, **(detection_profile or {})}

    def detect_keys(self):
        """Detect individual piano keys within the keyboard region"""
        if not self.keyboard_region:
            raise ValueError("Must detect keyboard region first")

        top_y, bottom_y, left_x, right_x = self.keyboard_region
        keyboard_img = self.image[top_y:bottom_y, left_x:right_x]
        keyboard_gray = cv2.cvtColor(keyboard_img, cv2.COLOR_BGR2GRAY)

        self.logger.debug(f"\n=== Detecting Keys in Region {right_x-left_x}x{bottom_y-top_y} ===")

        # Detect black keys first (easier to identify)
        self.black_keys = self._detect_black_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.black_keys)} black keys")

        self.logger.debug("First 5 black keys detected:")
        for i, (x, y, w, h) in enumerate(self.black_keys[:5]):
            self.logger.debug(f"  Black key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")

        # Detect white keys from black-key geometry by default.
        self.white_keys = self._detect_white_keys_from_black(keyboard_gray)
        # If geometry reconstruction under-detects badly, retry separator scan.
        if len(self.white_keys) < 4:
            self.white_keys = self._detect_white_keys(keyboard_gray)
        self.logger.debug(f"Detected {len(self.white_keys)} white keys")

        self.logger.debug("First 5 white keys detected:")
        for i, (x, y, w, h) in enumerate(self.white_keys[:5]):
            self.logger.debug(f"  White key {i}: x={x}, y={y}, w={w}, h={h} (absolute x={left_x + x})")

        self._maybe_recover_black_keys(keyboard_gray)

        return len(self.black_keys), len(self.white_keys)

    def run_complete_detection(self):
        """Run the complete detection pipeline"""
        self.logger.debug(f"\n{'='*60}")
        self.logger.debug(f"MONOLITHIC PIANO DETECTOR - Complete Analysis")
        self.logger.debug(f"{'='*60}")

        try:
            # Verify keyboard region was provided
            if not self.keyboard_region:
                raise ValueError("Keyboard region must be provided for manual ROI detection")

            self.logger.debug(f"Using provided keyboard region: {self.keyboard_region}")

            # Step 1: Detect individual keys
            num_black, num_white = self.detect_keys()

            # Step 2: Assign musical notes
            self.assign_notes()

            # Step 3: Create final visualization
            output_path = self.create_final_visualization()

            # Summary
            self.logger.debug(f"\n{'='*60}")
            self.logger.debug(f"DETECTION COMPLETE")
            self.logger.debug(f"{'='*60}")
            self.logger.debug(f"Keyboard region: {self.keyboard_region}")
            self.logger.debug(f"Black keys detected: {num_black}")
            self.logger.debug(f"White keys detected: {num_white}")
            self.logger.debug(f"Total keys: {num_black + num_white}")
            self.logger.debug(f"Notes assigned: {len(self.key_notes)}")
            self.logger.debug(f"Final result: {output_path}")

            return {
                'region': self.keyboard_region,
                'black_keys': num_black,
                'white_keys': num_white,
                'total_keys': num_black + num_white,
                'notes_assigned': len(self.key_notes),
                'output_path': output_path
            }

        except Exception as e:
            self.logger.debug(f"Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None


__all__ = ["DEFAULT_DETECTION_PARAMS", "MonolithicPianoDetector"]


if __name__ == "__main__":
    # Example usage - update path as needed
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Usage: python monolithic_detector.py <image_path>")
        print("Note: This detector requires a manual keyboard region (ROI).")
        sys.exit(1)

    # Manual ROI required - example coordinates (adjust as needed)
    # Format: (top_y, bottom_y, left_x, right_x)
    manual_roi = (100, 300, 50, 1850)  # Example values

    detector = MonolithicPianoDetector(image_path, keyboard_region=manual_roi)
    results = detector.run_complete_detection()
