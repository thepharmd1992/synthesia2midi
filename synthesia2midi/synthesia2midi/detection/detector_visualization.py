"""Visualization output for the manual piano keyboard detector."""

import os

import cv2

from synthesia2midi.runtime_paths import detect_runtime_paths


class DetectorVisualizationMixin:
    def create_final_visualization(self):
        """Create the final detection visualization on the full image"""
        if not self.keyboard_region or not self.key_notes:
            raise ValueError("Must complete detection and note assignment first")

        self.logger.debug(f"\n=== Creating Final Visualization ===")

        top_y, bottom_y, left_x, right_x = self.keyboard_region

        # Create labeled keyboard region
        keyboard_img = self.image[top_y:bottom_y, left_x:right_x].copy()

        # Draw key overlays and labels
        for center_x, note_info in self.key_notes.items():
            box = note_info['box']
            note = note_info['note']
            key_type = note_info['type']

            x, y, w, h = box

            # Draw bounding box
            color = (0, 255, 0) if key_type == 'white' else (0, 0, 255)
            cv2.rectangle(keyboard_img, (x, y), (x + w, y + h), color, 2)

            # Add note label with better positioning and visibility
            if key_type == 'white':
                # Place label at bottom of white key area, within the key region
                label_y = y + h - 5  # Near bottom of white key
                label_x = x + w // 2 - 10  # Center horizontally
                text_color = (255, 0, 0)  # Red text for better visibility on white
            else:
                # Place label in middle of black key
                label_y = y + h // 2 + 5  # Middle of black key
                label_x = x + w // 2 - 8  # Center horizontally
                text_color = (255, 255, 255)  # White text for visibility on black

            cv2.putText(keyboard_img, note, (max(0, label_x), label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Create final full image
        final_image = self.image.copy()
        final_image[top_y:bottom_y, left_x:right_x] = keyboard_img

        # Add region boundary
        cv2.rectangle(final_image, (left_x, top_y), (right_x, bottom_y), (0, 255, 0), 3)

        # Add title and stats
        cv2.putText(final_image, "Piano Keyboard Auto-Detection",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        stats = f"Region: y={top_y}-{bottom_y}, x={left_x}-{right_x} | Keys: {len(self.black_keys)} black, {len(self.white_keys)} white"
        cv2.putText(final_image, stats, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Save final result
        output_dir = str(detect_runtime_paths().debug_dir())
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'final_detection_result.jpg')
        cv2.imwrite(output_path, final_image)
        self.logger.debug(f"Final detection saved to: {output_path}")

        return output_path
