# NeoDraw

Air drawing and live object detection on a webcam, with the object detector running on the **AMD Ryzen AI NPU**.

- **YOLOv8m** (quantized, XINT8) runs on the NPU through ONNX Runtime's `VitisAIExecutionProvider`.
- **MediaPipe hand tracking** runs on the CPU and turns hand shapes into drawing gestures.
- Detection runs in a background thread, so the view stays smooth while the NPU works.

Measured on a Ryzen 7 7840HS (Phoenix NPU, 1280x720 camera):

| Stage | Where | Time |
|---|---|---|
| YOLOv8m detection | NPU | about 60 ms per frame (13 to 14 detections/s) |
| Hand tracking | CPU | about 17 ms per frame |
| Display loop | | about 26 FPS |

## Gestures

| Hand | Action |
|---|---|
| Index finger only, thumb tucked | Draw |
| Index + middle | Next colour (once per gesture) |
| Index + middle + ring | Next brush width: 4, 8, 14, 22, 32 px (once per gesture) |
| All five fingers open | Erase around the palm |
| Thumb + index | Zoom: spread to zoom in, pinch to zoom out (1x to 4x) |

Keys: `c` clears the drawing, `z` resets zoom, `q` or `Esc` quits.

## Requirements

- Windows 11 on a Ryzen AI processor with an NPU. Tested only on **Phoenix** (Ryzen 7040 series).
- The AMD NPU driver (check with `xrt-smi examine`).
- [Ryzen AI Software 1.7.0](https://ryzenai.docs.amd.com/en/latest/inst.html). It is not redistributed here.
- [uv](https://docs.astral.sh/uv/) and PowerShell 7.4 or newer.

## Setup

```powershell
git clone https://github.com/SilentDemonSD/NeoDraw
cd NeoDraw
.\setup.ps1
```

`setup.ps1` finds the SDK through `RYZEN_AI_INSTALLATION_PATH` (set by the AMD installer), or pass it explicitly:

```powershell
.\setup.ps1 -Sdk "C:\Program Files\RyzenAI\1.7.0"
```

It then:

1. creates a Python 3.12 virtual environment in `.venv`,
2. installs AMD's `onnxruntime-vitisai` and `voe` wheels from the SDK plus `requirements.txt`,
3. copies the Vitis AI provider DLLs into `onnxruntime\capi` (uv leaves them in the wheel's data folder, and the NPU session crashes without them),
4. downloads the models (about 110 MB) into `models/`,
5. records the SDK path in `neodraw.toml`,
6. runs the tests and the hardware check.

The first run compiles the model for the NPU and takes up to a minute. The result is cached in `cache/`, so later starts take a couple of seconds.

## Run

```powershell
.\run.bat                         # start the app
.\run.bat --camera "HP True"      # pick a camera by part of its name
.\run.bat check                   # NPU + camera + hand tracker check with timings
.\run.bat fetch                   # download any missing models
.venv\Scripts\python.exe -m tests.test_interaction   # gesture, canvas and zoom tests (no hardware)
```

## Configuration

Copy `neodraw.example.toml` to `neodraw.toml` (ignored by git) and change what you need:

| Key | Default | Meaning |
|---|---|---|
| `camera` | `""` | Part of the camera name. Empty picks the first camera. |
| `width`, `height` | `1280`, `720` | Capture size |
| `sdk` | `RYZEN_AI_INSTALLATION_PATH` | Ryzen AI SDK folder |
| `npu_target` | `RyzenAI_vision_config_2` | `vaip_config.json` target |
| `npu_overlay` | `phoenix/4x4.xclbin` | NPU overlay, relative to `voe-4.0-win_amd64\xclbins` |
| `score`, `iou` | `0.45`, `0.5` | Detection confidence and NMS thresholds |
| `max_zoom` | `4.0` | Zoom limit |
| `eraser_radius` | `60` | Eraser size in pixels |
| `hold_frames` | `3` | Frames a gesture must hold before it takes effect |

### Running on the NPU, not the CPU

The provider silently falls back to the CPU when the target does not match the chip, and still reports `VitisAIExecutionProvider`. On Phoenix:

- `target: RyzenAI_vision_config_2` with `phoenix/4x4.xclbin` puts all but 3 layers on the NPU.
- The documented `target: X1` leaves 264 layers on the CPU.

Use `.\run.bat check` and Task Manager's NPU graph to confirm. Newer chips (Strix, Krackan) need a different target and overlay; those are untested here.

## Layout

```
neodraw/
  config.py     Settings, loaded from neodraw.toml
  assets.py     model downloads
  runtime.py    NPU session factory (Vitis AI provider options, compile cache)
  detector.py   YOLOv8m pre/post-processing, background detection thread
  gestures.py   hand shape to gesture (pure logic)
  hands.py      MediaPipe hand tracker
  canvas.py     drawing, erasing, colour and width
  zoom.py       pinch zoom
  camera.py     camera selection by name
  app.py        the main loop and overlay
  check.py      hardware check
tests/          hardware-free tests
setup.ps1       one-step install
```

To add another NPU model, build its session with `NpuRuntime.session(model, cache_key)` and give it a `detect(frame)` method; `BackgroundDetector` runs anything with that shape.

## Why not Mojo?

Mojo has no native Windows build (WSL only), and Modular's MAX runtime targets CPUs and GPUs, not the AMD XDNA NPU. The only supported route to this NPU is AMD's Vitis AI execution provider, which is driven from Python or C++. The heavy work already runs in native code (NPU, OpenCV, MediaPipe), so Python is not the bottleneck.

## Models

Models are downloaded at setup, not stored in this repository:

- [amd/yolov8m](https://huggingface.co/amd/yolov8m): quantized YOLOv8m for Ryzen AI
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

Check each model's licence before using it in your own project.
