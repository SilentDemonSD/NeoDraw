import math

FINGER_JOINTS = ((8, 6), (12, 10), (16, 14), (20, 18))
PALM = (0, 5, 9, 13, 17)
SHAPES = {(1, 1, 0, 0): "color", (1, 1, 1, 0): "width"}


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Gestures:
    @staticmethod
    def fingers(points):
        thumb = distance(points[4], points[9]) > distance(points[3], points[9])
        others = [distance(points[tip], points[0]) > distance(points[pip], points[0]) for tip, pip in FINGER_JOINTS]
        return (int(thumb), *map(int, others))

    @classmethod
    def classify(cls, points):
        thumb, *others = cls.fingers(points)
        others = tuple(others)
        if thumb and others == (1, 1, 1, 1):
            return "erase"
        if others == (1, 0, 0, 0):
            return "zoom" if thumb else "draw"
        return SHAPES.get(others, "idle")

    @staticmethod
    def pinch(points):
        return distance(points[4], points[8]) / max(distance(points[0], points[9]), 1.0)
