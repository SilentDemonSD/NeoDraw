import numpy as np

from neodraw.canvas import AirCanvas
from neodraw.gestures import FINGER_JOINTS, Gestures
from neodraw.zoom import DigitalZoom


def fake_hand(thumb_up, fingers_up, pinch=1.0):
    points = [(0.0, 0.0)] * 21
    points[9] = (0.0, 4.0)
    points[3] = (-3.0, 4.0)
    points[4] = (-6.0 * pinch, 4.0) if thumb_up else (-1.0, 4.0)
    for (tip, pip), up in zip(FINGER_JOINTS, fingers_up):
        points[pip] = (0.0, 5.0)
        points[tip] = (0.0, 10.0 if up else 3.0)
    return points


def shifted(points):
    return [(x + 100, y + 100) for x, y in points]


def test_gestures():
    cases = (((0, (1, 0, 0, 0)), "draw"), ((1, (1, 0, 0, 0)), "zoom"), ((0, (1, 1, 0, 0)), "color"),
             ((1, (1, 1, 1, 0)), "width"), ((1, (1, 1, 1, 1)), "erase"), ((0, (1, 1, 1, 1)), "idle"),
             ((0, (0, 0, 0, 0)), "idle"))
    for (thumb, fingers), expected in cases:
        assert Gestures.classify(fake_hand(thumb, fingers)) == expected, (thumb, fingers, expected)


def test_canvas():
    canvas = AirCanvas((200, 200, 3), hold_frames=2)
    open_hand, pointing = shifted(fake_hand(1, (1, 1, 1, 1))), shifted(fake_hand(0, (1, 0, 0, 0)))
    for gesture, expected in (("color", (1, 1)), ("width", (1, 2))):
        for _ in range(2):
            canvas.update(gesture, pointing)
        assert (canvas.color_index, canvas.width_index) == expected, gesture
    for _ in range(3):
        canvas.update("draw", pointing)
    assert canvas.ink.any()
    for _ in range(6):
        canvas.update("erase", open_hand)
    assert not canvas.ink.any()


def test_zoom():
    zoom = DigitalZoom(smoothing=1.0)
    zoom.update(True, fake_hand(1, (1, 0, 0, 0), pinch=1.0))
    zoom.update(True, fake_hand(1, (1, 0, 0, 0), pinch=2.0))
    assert zoom.level > 1.5, zoom.level
    assert zoom.apply(np.zeros((72, 128, 3), np.uint8)).shape == (72, 128, 3)
    zoom.update(False, [])
    zoom.update(True, fake_hand(1, (1, 0, 0, 0), pinch=2.0))
    zoom.update(True, fake_hand(1, (1, 0, 0, 0), pinch=0.5))
    assert zoom.level == 1.0, zoom.level


if __name__ == "__main__":
    test_gestures()
    test_canvas()
    test_zoom()
    print("interaction tests ok")
