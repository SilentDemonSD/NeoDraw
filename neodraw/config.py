import os
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    sdk: Path = Path(os.environ.get("RYZEN_AI_INSTALLATION_PATH", ROOT / "sdk" / "PFiles64" / "RyzenAI" / "1.7.0"))
    camera: str = ""
    width: int = 1280
    height: int = 720
    detector: str = "yolox-s"
    npu_target: str = "RyzenAI_vision_config_2"
    npu_overlay: str = "phoenix/4x4.xclbin"
    score: float = 0.45
    iou: float = 0.5
    max_zoom: float = 8.0
    depth_gain: float = 1.5
    eraser_radius: int = 60
    hold_frames: int = 3

    @classmethod
    def load(cls, path=ROOT / "neodraw.toml", **overrides):
        values = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        values.update({key: value for key, value in overrides.items() if value is not None})
        unknown = sorted(set(values) - {field.name for field in fields(cls)})
        if unknown:
            raise SystemExit(f"unknown settings in {path.name}: {unknown}")
        settings = cls(**values)
        return replace(settings, root=Path(settings.root), sdk=Path(settings.sdk))

    @property
    def xclbin(self):
        return self.sdk / "voe-4.0-win_amd64" / "xclbins" / self.npu_overlay

    @property
    def cache(self):
        return self.root / "cache"

    @property
    def hand_model(self):
        return self.root / "models" / "hand" / "hand_landmarker.task"

    @property
    def sample(self):
        return self.root / "samples" / "demo.jpg"
