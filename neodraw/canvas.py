import math
from collections import deque

import cv2
import numpy as np

from .gestures import PALM, Gestures

PALETTE = ((0, 0, 255), (0, 220, 0), (255, 140, 0), (0, 230, 255), (255, 0, 255), (255, 255, 255))
BRUSHES = (4, 8, 14, 22, 32)


class View:
    def __init__(self, shape, focal=900.0, min_zoom=0.25, max_zoom=8.0):
        height, width = shape[:2]
        self.centre = np.array([width / 2, height / 2], np.float32)
        self.focal, self.min_zoom, self.max_zoom = focal, min_zoom, max_zoom
        self.reset()

    def reset(self):
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self.pan = np.zeros(2, np.float32)

    @property
    def state(self):
        return self.yaw, self.pitch, self.zoom, float(self.pan[0]), float(self.pan[1])

    @property
    def rotation(self):
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
        return pitch @ yaw

    def perspective(self, depth):
        return self.focal / np.maximum(self.focal - depth, self.focal * 0.1)

    def project(self, points):
        view = points @ self.rotation.T * self.zoom
        scale = self.perspective(view[:, 2])
        return self.centre + self.pan + view[:, :2] * scale[:, None], scale

    def unproject(self, screen, depth):
        scale = self.perspective(depth)
        flat = (np.asarray(screen, np.float32) - self.centre - self.pan) / scale
        return np.array([flat[0], flat[1], depth], np.float32) / self.zoom @ self.rotation

    def rotate(self, dx, dy, speed=0.008):
        self.yaw += dx * speed
        self.pitch = float(np.clip(self.pitch - dy * speed, -1.4, 1.4))

    def zoom_about(self, anchor, level):
        level = float(np.clip(level, self.min_zoom, self.max_zoom))
        offset = np.asarray(anchor, np.float32) - self.centre - self.pan
        self.pan = self.pan + offset * (1 - level / self.zoom)
        self.zoom = level


class AirCanvas:
    def __init__(self, shape, hold_frames=3, smoothing=0.45, eraser=60, history=20, depth_gain=1.5, max_zoom=8.0):
        self.height = shape[0]
        self.view = View(shape, max_zoom=max_zoom)
        self.ink = np.zeros(shape, np.uint8)
        self.mask = np.zeros(shape[:2], np.uint8)
        self.strokes, self.active = [], None
        self.history = deque(maxlen=history)
        self.hold_frames, self.smoothing, self.eraser, self.depth_gain = hold_frames, smoothing, eraser, depth_gain
        self.color_index, self.width_index = 0, 1
        self.mode, self.candidate, self.streak = "idle", "idle", 0
        self.cursor, self.last, self.anchor, self.saved = None, None, None, False
        self.palm = self.palm_reference = None
        self.rendered, self.dirty, self.still, self.settle_frames = None, True, None, 6
        self.handlers = {"draw": self._draw, "erase": self._erase, "pan": self._pan, "rotate": self._rotate, "zoom": self._zoom}

    @property
    def color(self):
        return PALETTE[self.color_index]

    @property
    def brush(self):
        return BRUSHES[self.width_index]

    @property
    def depth(self):
        if self.palm is None:
            return 0.0
        return self.depth_gain * (self.palm / self.palm_reference - 1) * self.height

    def _track_depth(self, points):
        if not points:
            return
        palm = Gestures.palm_size(points)
        self.palm = palm if self.palm is None else 0.7 * self.palm + 0.3 * palm
        if self.palm_reference is None:
            self.palm_reference = palm
        elif self.mode != "draw":
            self.palm_reference += 0.02 * (palm - self.palm_reference)

    def _settle(self, gesture):
        self.streak = self.streak + 1 if gesture == self.candidate else 1
        self.candidate = gesture
        if self.streak < self.hold_frames or gesture == self.mode:
            return
        self._finish()
        self.mode, self.cursor, self.last, self.anchor, self.saved = gesture, None, None, None, False
        if gesture == "color":
            self.color_index = (self.color_index + 1) % len(PALETTE)
        if gesture == "width":
            self.width_index = (self.width_index + 1) % len(BRUSHES)

    def _checkpoint(self):
        self.history.append(tuple(self.strokes))

    def _checkpoint_once(self):
        if not self.saved:
            self._checkpoint()
            self.saved = True

    def _follow(self, target):
        target = np.asarray(target, np.float32)
        self.cursor = target if self.cursor is None else self.cursor + self.smoothing * (target - self.cursor)
        return self.cursor

    def _palm_centre(self, points):
        return np.mean([points[i] for i in PALM], axis=0)

    def update(self, gesture, points):
        self._track_depth(points)
        self._settle(gesture)
        handler = self.handlers.get(self.mode)
        if gesture != self.mode or handler is None or not points:
            self._finish()
            self.cursor = self.last = self.anchor = None
            return
        handler(points)

    def _active_stroke(self):
        points, color, width = self.active
        return np.array(points, np.float32), color, width

    def _finish(self):
        if self.active is not None:
            self.strokes.append(self._active_stroke())
            self.active = None

    def _draw(self, points):
        self._checkpoint_once()
        position = self._follow(points[8])
        screen = tuple(int(v) for v in position)
        if self.last is not None and math.dist(screen, self.last) < 1.5:
            return
        depth = self.depth
        world = self.view.unproject(position, depth)
        if self.active is None:
            width = self.brush / (self.view.zoom * float(self.view.perspective(depth)))
            self.active = ([world], self.color, width)
        else:
            self.active[0].append(world)
        start = self.last or screen
        cv2.line(self.ink, start, screen, self.color, self.brush, cv2.LINE_AA)
        cv2.line(self.mask, start, screen, 255, self.brush, cv2.LINE_AA)
        self.last = screen

    @staticmethod
    def _runs(keep):
        edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(np.int8), 0]))
        return zip(edges[::2], edges[1::2])

    def _erase(self, points):
        self._checkpoint_once()
        centre = self._follow(self._palm_centre(points))
        survivors, changed = [], False
        for stroke in self.strokes:
            screen, _ = self.view.project(stroke[0])
            keep = np.hypot(*(screen - centre).T) > self.eraser
            if keep.all():
                survivors.append(stroke)
                continue
            changed = True
            survivors += [(stroke[0][start:end], *stroke[1:]) for start, end in self._runs(keep) if end - start > 1]
        if changed:
            self.strokes, self.dirty = survivors, True

    def _motion(self, points):
        position = self._follow(self._palm_centre(points)).copy()
        delta = np.zeros(2, np.float32) if self.anchor is None else position - self.anchor
        self.anchor = position
        return delta

    def _pan(self, points):
        self.view.pan = self.view.pan + self._motion(points)

    def _rotate(self, points):
        self.view.rotate(*self._motion(points))

    def _zoom(self, points):
        pinch = Gestures.pinch(points)
        if self.anchor is None:
            self.anchor = (pinch, self.view.zoom, (np.array(points[4]) + np.array(points[8])) / 2)
        base_pinch, base_zoom, centre = self.anchor
        target = base_zoom * pinch / max(base_pinch, 1e-3)
        self.view.zoom_about(centre, self.view.zoom + self.smoothing * (target - self.view.zoom))

    def undo(self):
        if not self.history:
            return False
        self._finish()
        self.strokes = list(self.history.pop())
        self.cursor = self.last = None
        self.saved, self.dirty = False, True
        return True

    def clear(self):
        self._finish()
        self._checkpoint()
        self.strokes, self.dirty = [], True

    def _redraw(self, line_type):
        self.ink[:] = 0
        strokes = self.strokes + ([self._active_stroke()] if self.active is not None else [])
        if strokes:
            screen, scale = self.view.project(np.concatenate([stroke[0] for stroke in strokes]))
            groups, start = {}, 0
            for points, color, width in strokes:
                end = start + len(points)
                path = np.round(screen[start:end]).astype(np.int32)
                thickness = int(np.clip(round(width * self.view.zoom * float(scale[start:end].mean())), 1, 200))
                if len(path) == 1:
                    cv2.circle(self.ink, tuple(int(v) for v in path[0]), max(thickness // 2, 1), color, -1, line_type)
                else:
                    groups.setdefault((color, thickness), []).append(path)
                start = end
            for (color, thickness), paths in groups.items():
                cv2.polylines(self.ink, paths, False, color, thickness, line_type)
        cv2.threshold(cv2.cvtColor(self.ink, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY, dst=self.mask)
        self.rendered, self.dirty = self.view.state, False

    def render(self, frame):
        if self.dirty or self.rendered != self.view.state:
            self._redraw(cv2.LINE_8)
            self.still = 0
        elif self.still is not None:
            self.still += 1
            if self.still >= self.settle_frames:
                self._redraw(cv2.LINE_AA)
                self.still = None
        cv2.copyTo(self.ink, self.mask, frame)
        if self.cursor is None:
            return
        position = tuple(int(v) for v in self.cursor)
        if self.mode == "erase":
            cv2.circle(frame, position, self.eraser, (255, 255, 255), 2)
        elif self.mode == "draw":
            cv2.circle(frame, position, self.brush // 2 + 4, self.color, -1)

    def render_axes(self, frame, origin, length=36):
        for axis, color, name in ((0, (0, 0, 255), "x"), (1, (0, 220, 0), "y"), (2, (255, 140, 0), "z")):
            tip = self.view.rotation[:2, axis] * length
            end = (int(origin[0] + tip[0]), int(origin[1] + tip[1]))
            cv2.line(frame, origin, end, color, 2, cv2.LINE_AA)
            cv2.putText(frame, name, end, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
