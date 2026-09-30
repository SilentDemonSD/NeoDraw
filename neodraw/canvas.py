import cv2
import numpy as np

from .gestures import PALM

PALETTE = ((0, 0, 255), (0, 220, 0), (255, 140, 0), (0, 230, 255), (255, 0, 255), (255, 255, 255))
BRUSHES = (4, 8, 14, 22, 32)


class AirCanvas:
    def __init__(self, shape, hold_frames=3, smoothing=0.45, eraser=60):
        self.ink = np.zeros(shape, np.uint8)
        self.hold_frames, self.smoothing, self.eraser = hold_frames, smoothing, eraser
        self.color_index, self.width_index = 0, 1
        self.mode, self.candidate, self.streak = "idle", "idle", 0
        self.cursor, self.last = None, None

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
        self.mode, self.cursor, self.last = gesture, None, None
        if gesture == "color":
            self.color_index = (self.color_index + 1) % len(PALETTE)
        if gesture == "width":
            self.width_index = (self.width_index + 1) % len(BRUSHES)

    def update(self, gesture, points):
        self._settle(gesture)
        if gesture != self.mode or self.mode not in ("draw", "erase") or not points:
            self.cursor = self.last = None
            return
        target = np.mean([points[i] for i in PALM], axis=0) if self.mode == "erase" else np.array(points[8])
        self.cursor = target if self.cursor is None else self.cursor + self.smoothing * (target - self.cursor)
        position = tuple(int(v) for v in self.cursor)
        if self.mode == "erase":
            cv2.circle(self.ink, position, self.eraser, (0, 0, 0), -1)
            return
        cv2.line(self.ink, self.last or position, position, self.color, self.brush, cv2.LINE_AA)
        self.last = position

    def clear(self):
        self.ink[:] = 0

    def render(self, frame):
        painted = self.ink.any(axis=2)
        frame[painted] = self.ink[painted]
        if self.cursor is None:
            return
        position = tuple(int(v) for v in self.cursor)
        if self.mode == "erase":
            cv2.circle(frame, position, self.eraser, (255, 255, 255), 2)
        else:
            cv2.circle(frame, position, self.brush // 2 + 4, self.color, -1)
