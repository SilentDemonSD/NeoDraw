import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .detector import DETECTORS

HUB = "https://huggingface.co/amd/{}/resolve/main/"
HAND = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"


class Assets:
    def __init__(self, settings, detectors=None):
        self.sources = {
            settings.hand_model: HAND + "hand_landmarker.task",
            settings.sample: HUB.format("yolov5s") + "demo.jpg",
        }
        for key in detectors or (settings.detector,):
            detector = Assets.detector(key)
            self.sources.update({detector.model_dir(settings) / name: HUB.format(key) + name for name in detector.FILES})

    @staticmethod
    def detector(key):
        if key not in DETECTORS:
            raise SystemExit(f"unknown detector '{key}'; choose one of {sorted(DETECTORS)}")
        return DETECTORS[key]

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
