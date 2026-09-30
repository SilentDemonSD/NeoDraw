import cv2
import numpy as np

from .gestures import Gestures


class DigitalZoom:
    def __init__(self, max_level=4.0, smoothing=0.3):
        self.max_level, self.smoothing = max_level, smoothing
        self.level = self.target = 1.0
        self.anchor = None

    def update(self, active, points):
        if not active:
            self.anchor = None
        else:
            pinch = Gestures.pinch(points)
            if self.anchor is None:
                self.anchor = (pinch, self.target)
            base_pinch, base_level = self.anchor
            self.target = float(np.clip(base_level * pinch / max(base_pinch, 1e-3), 1.0, self.max_level))
        self.level += self.smoothing * (self.target - self.level)

    def reset(self):
        self.level = self.target = 1.0
        self.anchor = None

    def apply(self, frame):
        if self.level < 1.01:
            return frame
        height, width = frame.shape[:2]
        crop_h, crop_w = int(height / self.level), int(width / self.level)
        top, left = (height - crop_h) // 2, (width - crop_w) // 2
        return cv2.resize(frame[top:top + crop_h, left:left + crop_w], (width, height), interpolation=cv2.INTER_LINEAR)
