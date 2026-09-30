import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from .gestures import Gestures


class HandTracker:
    def __init__(self, settings):
        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(settings.hand_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ))
        self.clock = 0

    def track(self, frame):
        self.clock = max(self.clock + 1, int(time.perf_counter() * 1000))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.landmarker.detect_for_video(image, self.clock)
        if not result.hand_landmarks:
            return "none", []
        height, width = frame.shape[:2]
        points = [(p.x * width, p.y * height) for p in result.hand_landmarks[0]]
        return Gestures.classify(points), points

    def close(self):
        self.landmarker.close()
