"""
modules/hand_detector.py
------------------------
Detects hands in a frame using MediaPipe and returns:
  - Annotated frame (with hand landmarks drawn)
  - Flat landmark array (42 values: 21 keypoints × x,y)
  - Boolean: hand found or not

Install:
    pip install mediapipe opencv-python
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


class HandDetector:
    """
    Wraps MediaPipe Hands for single-hand detection.
    Returns landmarks as a flat numpy array for the gesture recognizer.
    """

    def __init__(self, min_detection_confidence: float = 0.75,
                 min_tracking_confidence: float = 0.75):
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            )
        self.mp_hands    = mp.solutions.hands
        self.mp_drawing  = mp.solutions.drawing_utils
        self.mp_styles   = mp.solutions.drawing_styles
        self.hands       = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> tuple:
        """
        Process a BGR frame.

        Returns
        -------
        annotated   : np.ndarray  — frame with landmarks drawn
        landmarks   : np.ndarray | None — flat array of 42 floats (21 × xy)
        hand_found  : bool
        """
        annotated  = frame.copy()
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = self.hands.process(rgb)

        landmarks  = None
        hand_found = False

        if results.multi_hand_landmarks:
            hand_found = True
            hl = results.multi_hand_landmarks[0]

            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                annotated, hl, self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            # Flatten to 42-element array, normalised to [0, 1]
            h, w = frame.shape[:2]
            coords = []
            for lm in hl.landmark:
                coords.extend([lm.x, lm.y])
            landmarks = np.array(coords, dtype=np.float32)

        return annotated, landmarks, hand_found

    def close(self):
        self.hands.close()
