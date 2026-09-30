import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

YOLO = "https://huggingface.co/amd/yolov8m/resolve/main/"
HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
SAMPLE = "https://huggingface.co/amd/yolov5s/resolve/main/"


class Assets:
    def __init__(self, settings):
        self.sources = {
            settings.yolo_dir / "yolov8m.onnx": YOLO + "yolov8m.onnx",
            settings.yolo_dir / "anchors.npy": YOLO + "anchors.npy",
            settings.yolo_dir / "strides.npy": YOLO + "strides.npy",
            settings.hand_model: HAND + "hand_landmarker.task",
            settings.sample: SAMPLE + "demo.jpg",
        }

    def missing(self):
        return [path for path in self.sources if not path.exists()]

    def _download(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        with urllib.request.urlopen(self.sources[path], timeout=60) as response, open(partial, "wb") as target:
            shutil.copyfileobj(response, target, 1 << 20)
        partial.replace(path)
        print(f"fetched {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    def fetch(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self._download, self.missing()))

    def require(self):
        missing = self.missing()
        if missing:
            raise SystemExit(f"missing {[path.name for path in missing]}; run: python -m neodraw fetch")
