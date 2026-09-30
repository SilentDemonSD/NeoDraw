import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


class Yolov8Detector:
    NAME = "YOLOv8m"

    def __init__(self, runtime, settings, size=640):
        self.size, self.score, self.iou = size, settings.score, settings.iou
        overlay = settings.npu_overlay.split("/")[0]
        self.session = runtime.session(settings.yolo_dir / "yolov8m.onnx", f"yolov8m_{overlay}")
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = np.load(settings.yolo_dir / "anchors.npy").T.astype(np.float32)
        self.strides = np.load(settings.yolo_dir / "strides.npy").T.astype(np.float32)
        self.bins = np.arange(16, dtype=np.float32)
        self.canvas = np.full((size, size, 3), 114, np.uint8)

    def _letterbox(self, frame):
        ratio = min(self.size / frame.shape[0], self.size / frame.shape[1])
        width, height = round(frame.shape[1] * ratio), round(frame.shape[0] * ratio)
        left, top = round((self.size - width) / 2 - 0.1), round((self.size - height) / 2 - 0.1)
        self.canvas[:] = 114
        self.canvas[top:top + height, left:left + width] = cv2.resize(frame, (width, height))
        tensor = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1 / 255)
        return tensor[None], ratio, np.array([left, top, left, top], np.float32)

    def detect(self, frame):
        tensor, ratio, offset = self._letterbox(frame)
        pred = np.concatenate([out.reshape(-1, 144) for out in self.session.run(None, {self.input_name: tensor})])
        labels = pred[:, 64:].argmax(1)
        confidence = 1 / (1 + np.exp(-pred[np.arange(len(pred)), 64 + labels]))
        keep = confidence > self.score
        bins = pred[keep, :64].reshape(-1, 4, 16)
        bins = np.exp(bins - bins.max(-1, keepdims=True))
        distance = (bins / bins.sum(-1, keepdims=True)) @ self.bins
        anchors, strides = self.anchors[keep], self.strides[keep]
        corners = np.concatenate([anchors - distance[:, :2], anchors + distance[:, 2:]], 1) * strides
        corners = (corners - offset) / ratio
        boxes = np.concatenate([corners[:, :2], corners[:, 2:] - corners[:, :2]], 1)
        picked = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), confidence[keep].tolist(), self.score, self.iou)).flatten()
        return [(boxes[i], float(confidence[keep][i]), CLASSES[labels[keep][i]]) for i in picked]


class BackgroundDetector:
    def __init__(self, detector):
        self.detector = detector
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.detections, self.latency_ms, self.rate = [], 0.0, 0.0
        self.finished = time.perf_counter()

    def _timed(self, frame):
        started = time.perf_counter()
        return self.detector.detect(frame), (time.perf_counter() - started) * 1000

    def poll(self, frame):
        if self.pending is None or self.pending.done():
            if self.pending is not None:
                self.detections, self.latency_ms = self.pending.result()
                now = time.perf_counter()
                self.rate, self.finished = 0.8 * self.rate + 0.2 / (now - self.finished), now
            self.pending = self.pool.submit(self._timed, frame.copy())
        return self.detections

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
