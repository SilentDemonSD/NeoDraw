from collections import deque

import cv2
import numpy as np

from .gestures import PALM

PALETTE = ((0, 0, 255), (0, 220, 0), (255, 140, 0), (0, 230, 255), (255, 0, 255), (255, 255, 255))
BRUSHES = (4, 8, 14, 22, 32)


class AirCanvas:
    def __init__(self, shape, hold_frames=3, smoothing=0.45, eraser=60, history=10):
        self.ink = np.zeros(shape, np.uint8)
        self.mask = np.zeros(shape[:2], np.uint8)
        self.history = deque(maxlen=history)
        self.hold_frames, self.smoothing, self.eraser = hold_frames, smoothing, eraser
        self.color_index, self.width_index = 0, 1
        self.mode, self.candidate, self.streak = "idle", "idle", 0
        self.cursor, self.last, self.saved = None, None, False

    @property
    def color(self):
        return PALETTE[self.color_index]

    @property
    def brush(self):
        return BRUSHES[self.width_index]

    def _settle(self, gesture):
        self.streak = self.streak + 1 if gesture == self.candidate else 1
        self.candidate = gesture
        if self.streak < self.hold_frames or gesture == self.mode:
            return
        self.mode, self.cursor, self.last, self.saved = gesture, None, None, False
        if gesture == "color":
            self.color_index = (self.color_index + 1) % len(PALETTE)
        if gesture == "width":
            self.width_index = (self.width_index + 1) % len(BRUSHES)

    def _checkpoint(self):
        self.history.append((self.ink.copy(), self.mask.copy()))

    def update(self, gesture, points):
        self._settle(gesture)
        if gesture != self.mode or self.mode not in ("draw", "erase") or not points:
            self.cursor = self.last = None
            return
        if not self.saved:
            self._checkpoint()
            self.saved = True
        target = np.mean([points[i] for i in PALM], axis=0) if self.mode == "erase" else np.array(points[8])
        self.cursor = target if self.cursor is None else self.cursor + self.smoothing * (target - self.cursor)
        position = tuple(int(v) for v in self.cursor)
        if self.mode == "erase":
            cv2.circle(self.ink, position, self.eraser, (0, 0, 0), -1)
            cv2.circle(self.mask, position, self.eraser, 0, -1)
            return
        start = self.last or position
        cv2.line(self.ink, start, position, self.color, self.brush, cv2.LINE_AA)
        cv2.line(self.mask, start, position, 255, self.brush, cv2.LINE_AA)
        self.last = position

    def undo(self):
        if not self.history:
            return False
        self.ink, self.mask = self.history.pop()
        self.cursor = self.last = None
        self.saved = False
        return True

    def clear(self):
        self._checkpoint()
        self.ink[:] = 0
        self.mask[:] = 0

    def render(self, frame):
        cv2.copyTo(self.ink, self.mask, frame)
        if self.cursor is None:
            return
        position = tuple(int(v) for v in self.cursor)
        if self.mode == "erase":
            cv2.circle(frame, position, self.eraser, (255, 255, 255), 2)
        else:
            cv2.circle(frame, position, self.brush // 2 + 4, self.color, -1)
