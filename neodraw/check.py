import time

import cv2

from .assets import Assets
from .camera import Camera
from .detector import Yolov8Detector
from .hands import HandTracker
from .runtime import NpuRuntime


class HardwareCheck:
    def __init__(self, settings):
        Assets(settings).require()
        self.settings = settings

    @staticmethod
    def _milliseconds(call, frame, runs=10):
        started = time.perf_counter()
        for _ in range(runs):
            call(frame)
        return (time.perf_counter() - started) * 1000 / runs

    def run(self):
        yolo = Yolov8Detector(NpuRuntime(self.settings), self.settings)
        labels = {label for _, _, label in yolo.detect(cv2.imread(str(self.settings.sample)))}
        assert {"car", "truck"} <= labels, labels
        camera = Camera(self.settings)
        frame = camera.read()
        camera.close()
        assert frame is not None, "camera returned no frame"
        hands = HandTracker(self.settings)
        yolo_ms, hand_ms = self._milliseconds(yolo.detect, frame), self._milliseconds(hands.track, frame)
        hands.close()
        print(f"check ok: sample={sorted(labels)}, camera={frame.shape}, "
              f"{yolo.NAME} {yolo_ms:.1f} ms/frame on {yolo.session.get_providers()[0]}, hands {hand_ms:.1f} ms/frame")
