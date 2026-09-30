import time

import cv2

from .assets import Assets
from .camera import Camera
from .canvas import AirCanvas
from .captures import Captures
from .detector import BackgroundDetector
from .hands import HandTracker
from .runtime import NpuRuntime

ARROWS = {2424832: (-10, 0), 2555904: (10, 0), 2490368: (0, -10), 2621440: (0, 10)}


class NeoDrawApp:
    TITLE = "NeoDraw  |  Ryzen AI NPU"
    HELP = "1 draw | 2 color | 3 width | 5 erase | pinch zoom | fist move | pinky rotate"
    KEYS = "u undo  c clear  z reset view  o orbit  arrows rotate  +/- zoom  s snap  r rec  d detect  q quit"

    def __init__(self, settings):
        Assets(settings).require()
        self.camera = Camera(settings)
        frame = self.camera.read()
        if frame is None:
            raise SystemExit("camera returned no frame")
        detector = Assets.detector(settings.detector)(NpuRuntime(settings), settings)
        self.detector_name = detector.NAME
        self.detector = BackgroundDetector(detector)
        self.hands = HandTracker(settings)
        self.canvas = AirCanvas(frame.shape, settings.hold_frames, eraser=settings.eraser_radius,
                                depth_gain=settings.depth_gain, max_zoom=settings.max_zoom)
        self.captures = Captures(settings.root / "captures")
        self.detecting, self.orbiting, self.fps = True, False, 0.0
        self.status, self.status_until = "", 0.0
        self.actions = {
            ord("u"): self.undo, ord("c"): self.clear, ord("s"): self.snapshot, ord("r"): self.record,
            ord("d"): self.toggle_detection, ord("z"): self.reset_view, ord("o"): self.toggle_orbit,
            ord("+"): lambda: self.zoom_by(1.25), ord("="): lambda: self.zoom_by(1.25), ord("-"): lambda: self.zoom_by(0.8),
        }

    def notify(self, message, seconds=2.5):
        self.status, self.status_until = message, time.perf_counter() + seconds
        print(message)

    @staticmethod
    def draw_boxes(frame, detections):
        for (x, y, w, h), confidence, label in detections:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 200, 255), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (int(x), max(int(y) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

    @staticmethod
    def draw_hand(frame, points, zooming):
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 3, (200, 200, 200), -1)
        if zooming:
            thumb, index = (tuple(int(v) for v in points[i]) for i in (4, 8))
            cv2.line(frame, thumb, index, (255, 255, 0), 3, cv2.LINE_AA)

    def draw_hud(self, frame):
        height, width = frame.shape[:2]
        view = self.canvas.view
        detection = (f"NPU {self.detector_name} {self.detector.rate:.0f}/s ({self.detector.latency_ms:.0f} ms)"
                     if self.detecting else "NPU detection off")
        cv2.rectangle(frame, (0, 0), (width, 64), (0, 0, 0), -1)
        cv2.putText(frame, f"{detection}   view {self.fps:.0f} FPS   zoom {view.zoom:.2f}x   "
                           f"yaw {view.yaw * 57.3:.0f}  pitch {view.pitch * 57.3:.0f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"mode: {self.canvas.mode}   width {self.canvas.brush}px   depth {self.canvas.depth:+.0f}   {self.HELP}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.rectangle(frame, (width - 60, 12), (width - 16, 52), self.canvas.color, -1)
        cv2.circle(frame, (width - 100, 32), self.canvas.brush // 2 + 1, self.canvas.color, -1)
        cv2.rectangle(frame, (0, height - 30), (width, height), (0, 0, 0), -1)
        footer = self.status if time.perf_counter() < self.status_until else self.KEYS
        cv2.putText(frame, footer, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        self.canvas.render_axes(frame, (width - 60, height - 80))
        if self.captures.recording:
            cv2.circle(frame, (width - 150, 32), 9, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width - 136, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def step(self, frame):
        frame = cv2.flip(frame, 1)
        detections = self.detector.poll(frame) if self.detecting else []
        gesture, points = self.hands.track(frame)
        self.canvas.update(gesture, points)
        if self.orbiting:
            self.canvas.view.rotate(3, 0)
        self.draw_boxes(frame, detections)
        self.draw_hand(frame, points, self.canvas.mode == "zoom" and gesture == "zoom")
        self.canvas.render(frame)
        return frame

    def undo(self):
        self.notify("undo" if self.canvas.undo() else "nothing to undo")

    def clear(self):
        self.canvas.clear()
        self.notify("cleared (u to undo)")

    def reset_view(self):
        self.canvas.view.reset()
        self.orbiting = False
        self.notify("view reset")

    def toggle_orbit(self):
        self.orbiting = not self.orbiting
        self.notify("orbit on" if self.orbiting else "orbit off")

    def zoom_by(self, factor):
        view = self.canvas.view
        view.zoom_about(view.centre, view.zoom * factor)

    def snapshot(self):
        self.notify(f"saved {self.captures.snapshot(self.frame).name}")

    def record(self):
        self.notify(self.captures.toggle_recording(self.frame, self.fps or 30))

    def toggle_detection(self):
        self.detecting = not self.detecting
        self.detector.detections = []
        self.notify("detection on" if self.detecting else "detection off (NPU idle)")

    def handle(self, key):
        if key in ARROWS:
            self.canvas.view.rotate(*ARROWS[key])
            return
        self.actions.get(key & 0xFF if key < 256 else key, lambda: None)()

    def run(self):
        last = time.perf_counter()
        try:
            while (frame := self.camera.read()) is not None:
                self.frame = self.step(frame)
                self.captures.write(self.frame)
                now = time.perf_counter()
                self.fps, last = 0.9 * self.fps + 0.1 / (now - last), now
                shown = self.frame.copy()
                self.draw_hud(shown)
                cv2.imshow(self.TITLE, shown)
                key = cv2.waitKeyEx(1)
                if key in (27, ord("q")):
                    break
                if key != -1:
                    self.handle(key)
        finally:
            self.close()

    def close(self):
        self.captures.close()
        self.detector.close()
        self.hands.close()
        self.camera.close()
        cv2.destroyAllWindows()
