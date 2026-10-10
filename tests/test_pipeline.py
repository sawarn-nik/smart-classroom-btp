import numpy as np

from src.pipeline.tracker import IoUTracker, iou


def test_iou():
    a = np.array([0, 0, 10, 10.])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30.])) == 0.0
    assert abs(iou(a, np.array([5, 0, 15, 10.])) - 50 / 150) < 1e-9


def test_tracker_keeps_ids_and_expires():
    tr = IoUTracker(max_missed=2)
    a, b = np.array([0, 0, 10, 10.]), np.array([50, 50, 60, 60.])
    ids = tr.update([a, b])
    assert ids == [0, 1]
    assert tr.update([b + 1, a + 1]) == [1, 0]  # order swapped, ids follow boxes
    for _ in range(3):
        tr.update([a])
    assert 1 not in tr.tracks and tr.update([b]) == [2]
