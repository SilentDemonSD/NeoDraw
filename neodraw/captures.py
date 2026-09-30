import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2


class Recorder:
    def __init__(self, path, shape, fps, backlog=90):
        self.path, self.dropped = path, 0
        height, width = shape[:2]
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")
        self.frames = queue.Queue(maxsize=backlog)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        while (frame := self.frames.get()) is not None:
            self.writer.write(frame)
        self.writer.release()

    def write(self, frame):
        try:
            self.frames.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1

    def close(self):
        self.frames.put(None)
        self.thread.join()


class Captures:
    def __init__(self, directory):
        self.directory = directory
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.recorder = None

    def _path(self, suffix):
        self.directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        return self.directory / f"neodraw-{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{int(now * 1000) % 1000:03d}{suffix}"

    @property
    def recording(self):
        return self.recorder is not None

    def snapshot(self, frame):
        path = self._path(".png")
        self.pool.submit(cv2.imwrite, str(path), frame.copy())
        return path

    def toggle_recording(self, frame, fps):
        if self.recorder is None:
            self.recorder = Recorder(self._path(".mp4"), frame.shape, max(round(fps), 5))
            return f"recording {self.recorder.path.name}"
        recorder, self.recorder = self.recorder, None
        recorder.close()
        return f"saved {recorder.path.name}" + (f" ({recorder.dropped} frames dropped)" if recorder.dropped else "")

    def write(self, frame):
        if self.recorder is not None:
            self.recorder.write(frame)

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        self.pool.shutdown(wait=True)
