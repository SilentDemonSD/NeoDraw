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


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


class YoloDetector:
    NAME = KEY = MODEL = ""
    CENTRED = True
    SIZE = 640

    def __init__(self, runtime, settings):
        self.score, self.iou = settings.score, settings.iou
        overlay = settings.npu_overlay.split("/")[0]
        self.session = runtime.session(self.model_dir(settings) / self.MODEL, f"{self.KEY}_{overlay}")
        self.input_name = self.session.get_inputs()[0].name
        self.canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        self.buffer = np.empty((1, self.SIZE, self.SIZE, 3), np.float32)
        self.threshold = float(np.log(self.score / (1 - self.score)))

    @classmethod
    def model_dir(cls, settings):
        return settings.root / "models" / cls.KEY

    def _letterbox(self, frame):
        ratio = min(self.SIZE / frame.shape[0], self.SIZE / frame.shape[1])
        width, height = round(frame.shape[1] * ratio), round(frame.shape[0] * ratio)
        left, top = (round((self.SIZE - width) / 2 - 0.1), round((self.SIZE - height) / 2 - 0.1)) if self.CENTRED else (0, 0)
        self.canvas[:] = 114
        self.canvas[top:top + height, left:left + width] = cv2.resize(frame, (width, height))
        return ratio, np.array([left, top, left, top], np.float32)

    def _tensor(self):
        raise NotImplementedError

    def _decode(self, pred):
        raise NotImplementedError

    def infer(self, frame):
        ratio, offset = self._letterbox(frame)
        outputs = self.session.run(None, {self.input_name: self._tensor()})
        return outputs, ratio, offset

    def detect(self, frame):
        outputs, ratio, offset = self.infer(frame)
        pred = np.concatenate([out.reshape(-1, out.shape[-1]) for out in outputs])
        corners, confidence, labels = self._decode(pred)
        corners = (corners - offset) / ratio
        boxes = np.concatenate([corners[:, :2], corners[:, 2:] - corners[:, :2]], 1)
        picked = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), confidence.tolist(), self.score, self.iou), int).flatten()
        return [(boxes[i], float(confidence[i]), CLASSES[labels[i]]) for i in picked]


class Yolov8Detector(YoloDetector):
    NAME, KEY, MODEL = "YOLOv8m", "yolov8m", "yolov8m.onnx"
    FILES = ("yolov8m.onnx", "anchors.npy", "strides.npy")

    def __init__(self, runtime, settings):
        super().__init__(runtime, settings)
        self.anchors = np.load(self.model_dir(settings) / "anchors.npy").T.astype(np.float32)
        self.strides = np.load(self.model_dir(settings) / "strides.npy").T.astype(np.float32)
        self.bins = np.arange(16, dtype=np.float32)

    def _tensor(self):
        np.multiply(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB), np.float32(1 / 255), out=self.buffer[0], casting="unsafe")
        return self.buffer

    def _decode(self, pred):
        labels = pred[:, 64:].argmax(1)
        confidence = sigmoid(pred[np.arange(len(pred)), 64 + labels])
        keep = confidence > self.score
        bins = pred[keep, :64].reshape(-1, 4, 16)
        bins = np.exp(bins - bins.max(-1, keepdims=True))
        distance = (bins / bins.sum(-1, keepdims=True)) @ self.bins
        anchors, strides = self.anchors[keep], self.strides[keep]
        corners = np.concatenate([anchors - distance[:, :2], anchors + distance[:, 2:]], 1) * strides
        return corners, confidence[keep], labels[keep]


class YoloxDetector(YoloDetector):
    NAME, KEY, MODEL = "YOLOX-S", "yolox-s", "yolox-s-int8.onnx"
    FILES = ("yolox-s-int8.onnx",)
    CENTRED = False

    def __init__(self, runtime, settings):
        super().__init__(runtime, settings)
        grids, strides = [], []
        for stride in (8, 16, 32):
            cells = self.SIZE // stride
            x, y = np.meshgrid(np.arange(cells), np.arange(cells))
            grids.append(np.stack((x, y), 2).reshape(-1, 2))
            strides.append(np.full((cells * cells, 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.stride = np.concatenate(strides).astype(np.float32)

    def _tensor(self):
        np.copyto(self.buffer[0], self.canvas, casting="unsafe")
        return self.buffer

    def _decode(self, pred):
        candidates = np.flatnonzero(pred[:, 4] > self.threshold)
        pred, grid, stride = pred[candidates], self.grid[candidates], self.stride[candidates]
        labels = pred[:, 5:].argmax(1)
        confidence = sigmoid(pred[:, 4]) * sigmoid(pred[np.arange(len(pred)), 5 + labels])
        keep = confidence > self.score
        centre = (pred[keep, :2] + grid[keep]) * stride[keep]
        half = np.exp(pred[keep, 2:4]) * stride[keep] / 2
        return np.concatenate([centre - half, centre + half], 1), confidence[keep], labels[keep]


DETECTORS = {detector.KEY: detector for detector in (Yolov8Detector, YoloxDetector)}


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
