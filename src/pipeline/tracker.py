"""Minimal IoU tracker for face boxes (classroom students barely move)."""
import numpy as np


def iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_thr=0.3, max_missed=15):
        self.iou_thr, self.max_missed = iou_thr, max_missed
        self.tracks = {}  # id -> {"box", "missed"}
        self.next_id = 0

    def update(self, boxes):
        """boxes: list of xyxy arrays. Returns list of track ids, same order."""
        ids = [None] * len(boxes)
        pairs = sorted(((iou(t["box"], b), tid, j) for tid, t in self.tracks.items()
                        for j, b in enumerate(boxes)), reverse=True)
        used_t, used_b = set(), set()
        for s, tid, j in pairs:
            if s < self.iou_thr:
                break
            if tid in used_t or j in used_b:
                continue
            used_t.add(tid); used_b.add(j); ids[j] = tid
            self.tracks[tid] = {"box": np.asarray(boxes[j]), "missed": 0}
        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        for j, b in enumerate(boxes):
            if ids[j] is None:
                ids[j] = self.next_id
                self.tracks[self.next_id] = {"box": np.asarray(b), "missed": 0}
                self.next_id += 1
        return ids
