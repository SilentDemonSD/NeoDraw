import time

import cv2

from .assets import Assets
from .camera import Camera
from .canvas import AirCanvas
from .detector import BackgroundDetector, Yolov8Detector
from .hands import HandTracker
from .runtime import NpuRuntime
from .zoom import DigitalZoom


class NeoDrawApp:
    TITLE = "NeoDraw  |  Ryzen AI NPU  (q quit, c clear, z reset zoom)"
    HELP = "1 draw | 2 color | 3 width | 5 erase | thumb+index zoom"

    def __init__(self, settings):
        Assets(settings).require()
        self.camera = Camera(settings)
        frame = self.camera.read()
        if frame is None:
            raise SystemExit("camera returned no frame")
        self.detector = BackgroundDetector(Yolov8Detector(NpuRuntime(settings), settings))
        self.hands = HandTracker(settings)
        self.zoom = DigitalZoom(settings.max_zoom)
        self.canvas = AirCanvas(frame.shape, settings.hold_frames, eraser=settings.eraser_radius)

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

    def draw_hud(self, frame, fps):
        width = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (width, 64), (0, 0, 0), -1)
        cv2.putText(frame, f"NPU {Yolov8Detector.NAME} {self.detector.rate:.0f}/s ({self.detector.latency_ms:.0f} ms)"
                           f"   view {fps:.0f} FPS   zoom {self.zoom.level:.1f}x",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"mode: {self.canvas.mode}   width {self.canvas.brush}px   {self.HELP}",
                    (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.rectangle(frame, (width - 60, 12), (width - 16, 52), self.canvas.color, -1)
        cv2.circle(frame, (width - 100, 32), self.canvas.brush // 2 + 1, self.canvas.color, -1)

    def step(self, frame):
        frame = self.zoom.apply(cv2.flip(frame, 1))
        detections = self.detector.poll(frame)
        gesture, points = self.hands.track(frame)
        self.canvas.update(gesture, points)
        zooming = self.canvas.mode == "zoom" and gesture == "zoom"
        self.zoom.update(zooming, points)
        self.draw_boxes(frame, detections)
        self.draw_hand(frame, points, zooming)
        self.canvas.render(frame)
        return frame

    def run(self):
        fps, last = 0.0, time.perf_counter()
        try:
            while (frame := self.camera.read()) is not None:
                frame = self.step(frame)
                now = time.perf_counter()
                fps, last = 0.9 * fps + 0.1 / (now - last), now
                self.draw_hud(frame, fps)
                cv2.imshow(self.TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("c"):
                    self.canvas.clear()
                if key == ord("z"):
                    self.zoom.reset()
        finally:
            self.close()

    def close(self):
        self.detector.close()
        self.hands.close()
        self.camera.close()
        cv2.destroyAllWindows()
