import math
import tempfile
from pathlib import Path

import numpy as np

from neodraw.canvas import AirCanvas, View
from neodraw.captures import Captures
from neodraw.gestures import FINGER_JOINTS, Gestures


def fake_hand(thumb_up, fingers_up, pinch=1.0, at=(0.0, 0.0), size=1.0):
    points = [(0.0, 0.0)] * 21
    points[9] = (0.0, 4.0)
    points[3] = (-3.0, 4.0)
    points[4] = (-6.0 * pinch, 4.0) if thumb_up else (-1.0, 4.0)
    for (tip, pip), up in zip(FINGER_JOINTS, fingers_up):
        points[pip] = (0.0, 5.0)
        points[tip] = (0.0, 10.0 if up else 3.0)
    return [(at[0] + x * size, at[1] + y * size) for x, y in points]


POINTING, OPEN, FIST, SHAKA = (0, (1, 0, 0, 0)), (1, (1, 1, 1, 1)), (0, (0, 0, 0, 0)), (1, (0, 0, 0, 1))


def run(canvas, gesture, shape, frames, at=(100.0, 100.0), step=(0.0, 0.0), **hand):
    for index in range(frames):
        position = (at[0] + step[0] * index, at[1] + step[1] * index)
        canvas.update(gesture, fake_hand(*shape, at=position, **hand))


def test_gestures():
    cases = (((0, (1, 0, 0, 0)), "draw"), ((1, (1, 0, 0, 0)), "zoom"), ((0, (1, 1, 0, 0)), "color"),
             ((1, (1, 1, 1, 0)), "width"), ((1, (1, 1, 1, 1)), "erase"), ((0, (1, 1, 1, 1)), "idle"),
             ((0, (0, 0, 0, 0)), "pan"), ((1, (0, 0, 0, 0)), "pan"), ((1, (0, 0, 0, 1)), "rotate"),
             ((0, (0, 0, 0, 1)), "rotate"), ((0, (0, 1, 0, 0)), "idle"))
    for (thumb, fingers), expected in cases:
        assert Gestures.classify(fake_hand(thumb, fingers)) == expected, (thumb, fingers, expected)


def test_view_round_trip():
    view = View((720, 1280, 3))
    view.yaw, view.pitch, view.zoom, view.pan = 0.6, -0.3, 2.5, np.array([40.0, -25.0], np.float32)
    for screen, depth in (((300.0, 200.0), 0.0), ((900.0, 500.0), 120.0), ((640.0, 360.0), -80.0)):
        world = view.unproject(screen, depth)
        projected, _ = view.project(world[None])
        assert np.allclose(projected[0], screen, atol=1e-2), (projected, screen)
    anchor = np.array([800.0, 300.0], np.float32)
    world = view.unproject(anchor, 0.0)
    view.zoom_about(anchor, 5.0)
    assert np.allclose(view.project(world[None])[0][0], anchor, atol=1e-2)
    view.zoom_about(anchor, 100.0)
    assert view.zoom == view.max_zoom


def test_draw_erase_and_depth():
    canvas = AirCanvas((200, 200, 3), hold_frames=2, smoothing=1.0, eraser=20)
    for gesture, expected in (("color", (1, 1)), ("width", (1, 2))):
        run(canvas, gesture, POINTING, 2)
        assert (canvas.color_index, canvas.width_index) == expected, gesture
    run(canvas, "idle", FIST, 2)
    run(canvas, "draw", POINTING, 12, at=(40.0, 60.0), step=(10.0, 0.0))
    run(canvas, "idle", FIST, 2)
    assert len(canvas.strokes) == 1 and len(canvas.strokes[0][0]) >= 10
    assert abs(canvas.strokes[0][0][:, 2]).max() < 1e-3
    run(canvas, "erase", OPEN, 4, at=(100.0, 55.0))
    run(canvas, "idle", FIST, 2)
    assert len(canvas.strokes) == 2, [len(s[0]) for s in canvas.strokes]
    run(canvas, "draw", POINTING, 8, at=(40.0, 150.0), step=(10.0, 0.0), size=1.4)
    run(canvas, "idle", FIST, 2)
    assert canvas.strokes[-1][0][:, 2].max() > 20, canvas.strokes[-1][0][:, 2]


def test_pan_rotate_zoom_and_render():
    canvas = AirCanvas((200, 200, 3), hold_frames=1, smoothing=1.0)
    run(canvas, "draw", POINTING, 8, at=(60.0, 90.0), step=(10.0, 0.0))
    run(canvas, "idle", FIST, 1)
    frame = np.full((200, 200, 3), 50, np.uint8)
    canvas.render(frame)
    before = canvas.mask.copy()
    painted = before > 0
    assert painted.any() and (frame[painted] == canvas.ink[painted]).all() and (frame[~painted] == 50).all()
    run(canvas, "pan", FIST, 5, at=(100.0, 100.0), step=(0.0, 6.0))
    assert np.allclose(canvas.view.pan, (0, 24)), canvas.view.pan
    canvas.render(frame)
    assert not np.array_equal(before, canvas.mask)
    run(canvas, "rotate", SHAKA, 5, step=(10.0, 0.0))
    assert math.isclose(canvas.view.yaw, 40 * 0.008, rel_tol=1e-4), canvas.view.yaw
    zoom = canvas.view.zoom
    run(canvas, "zoom", (1, (1, 0, 0, 0)), 1, pinch=1.0)
    run(canvas, "zoom", (1, (1, 0, 0, 0)), 3, pinch=2.0)
    assert canvas.view.zoom > zoom * 1.5, canvas.view.zoom


def test_undo_and_clear():
    canvas = AirCanvas((200, 200, 3), hold_frames=1, smoothing=1.0)
    run(canvas, "draw", POINTING, 5, at=(40.0, 60.0), step=(10.0, 0.0))
    run(canvas, "idle", FIST, 1)
    run(canvas, "draw", POINTING, 5, at=(40.0, 120.0), step=(10.0, 0.0))
    run(canvas, "idle", FIST, 1)
    assert len(canvas.strokes) == 2
    assert canvas.undo() and len(canvas.strokes) == 1
    canvas.clear()
    assert canvas.strokes == []
    assert canvas.undo() and len(canvas.strokes) == 1
    assert canvas.undo() and canvas.strokes == [] and not canvas.undo()


def test_captures():
    with tempfile.TemporaryDirectory() as directory:
        captures = Captures(Path(directory))
        frame = np.full((72, 128, 3), 90, np.uint8)
        image = captures.snapshot(frame)
        captures.toggle_recording(frame, 30)
        for _ in range(10):
            captures.write(frame)
        message = captures.toggle_recording(frame, 30)
        captures.close()
        videos = list(Path(directory).glob("*.mp4"))
        assert image.exists() and message.startswith("saved"), message
        assert len(videos) == 1 and videos[0].stat().st_size > 0


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("interaction tests ok")
