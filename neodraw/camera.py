import cv2
from cv2_enumerate_cameras import enumerate_cameras


class Camera:
    def __init__(self, settings):
        cameras = enumerate_cameras(cv2.CAP_MSMF)
        chosen = next((c for c in cameras if settings.camera.lower() in c.name.lower()), None)
        if chosen is None:
            raise SystemExit(f"no camera matching '{settings.camera}'; available: {[c.name for c in cameras]}")
        print(f"camera: {chosen.name} (index {chosen.index})")
        self.capture = cv2.VideoCapture(chosen.index, cv2.CAP_MSMF)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self):
        self.capture.release()
