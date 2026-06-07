"""
extractor.py — Extract hand keypoints from images/videos using MediaPipe.

Usage:
    extractor = KeypointExtractor()
    keypoints = extractor.extract_from_image('frame.jpg')  # [21, 3] or None
    extractor.process_dataset_folder('data/raw/ASL', 'data/keypoints/ASL')
"""

import cv2
import numpy as np
import mediapipe as mp
import os
from pathlib import Path
from tqdm import tqdm


class KeypointExtractor:
    """
    Wraps MediaPipe Hands to extract 21 3D landmarks per hand.

    Args:
        max_hands: max hands to detect (1 for single-hand gestures)
        min_detection_confidence: MediaPipe threshold
        min_tracking_confidence: MediaPipe tracking threshold
        prefer_right: if True and two hands detected, pick right hand
    """

    def __init__(
        self,
        max_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        prefer_right=True,
    ):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.prefer_right = prefer_right

    def extract_from_image(self, image_path_or_array):
        """
        Extract keypoints from a single image.

        Args:
            image_path_or_array: path string or numpy BGR array

        Returns:
            keypoints: np.ndarray [21, 3] (x, y, z) or None if no hand found
            handedness: 'Left' | 'Right' | None
        """
        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
            if frame is None:
                return None, None
        else:
            frame = image_path_or_array

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None, None

        # Pick the right hand if available and prefer_right is set
        hand_idx = 0
        handedness = 'Right'
        if result.multi_handedness and len(result.multi_handedness) > 1 and self.prefer_right:
            for i, hand_info in enumerate(result.multi_handedness):
                if hand_info.classification[0].label == 'Right':
                    hand_idx = i
                    break
            handedness = result.multi_handedness[hand_idx].classification[0].label

        landmarks = result.multi_hand_landmarks[hand_idx]
        keypoints = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
            dtype=np.float32
        )  # [21, 3]

        return keypoints, handedness

    def extract_from_video(self, video_path, sample_every_n=1):
        """
        Extract keypoints from every N-th frame of a video.

        Returns:
            List of (frame_idx, keypoints) tuples. Skips frames without hands.
        """
        cap = cv2.VideoCapture(video_path)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        results = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every_n == 0:
                kp, _ = self.extract_from_image(frame)
                if kp is not None:
                    results.append((frame_idx, kp))

            frame_idx += 1

        cap.release()
        return results

    def process_dataset_folder(self, input_dir, output_dir, extensions=('.jpg', '.jpeg', '.png')):
        """
        Batch-process an image folder. Saves .npy keypoint files
        mirroring the input folder structure.

        Args:
            input_dir: root folder with images, subfolders = class names
            output_dir: where to save .npy files
            extensions: image file types to process
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.rglob(f'*{ext}'))

        success, failed = 0, 0
        for img_path in tqdm(image_paths, desc=f'Extracting keypoints from {input_dir.name}'):
            kp, _ = self.extract_from_image(str(img_path))

            # Mirror folder structure
            rel = img_path.relative_to(input_dir)
            out_path = output_dir / rel.with_suffix('.npy')
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if kp is not None:
                np.save(str(out_path), kp)
                success += 1
            else:
                failed += 1

        print(f'Done: {success} extracted, {failed} failed (no hand detected).')
        return success, failed

    def visualize(self, image_path_or_array, keypoints=None):
        """Draw keypoints on image and return annotated frame (BGR)."""
        mp_draw = mp.solutions.drawing_utils

        if isinstance(image_path_or_array, str):
            frame = cv2.imread(image_path_or_array)
        else:
            frame = image_path_or_array.copy()

        if keypoints is None:
            keypoints, _ = self.extract_from_image(frame)

        if keypoints is not None:
            h, w = frame.shape[:2]
            for i, (x, y, z) in enumerate(keypoints):
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        return frame

    def __del__(self):
        self.hands.close()
