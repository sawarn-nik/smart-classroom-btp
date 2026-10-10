"""Unified smart-classroom pipeline: one face detector feeds attendance and
engagement; a YOLO detector gives behaviour boxes that are linked to students.

Per frame:
  1. SCRFD face detection (shared)            -> face boxes + 5 landmarks
  2. IoU tracking                              -> stable track ids
  3. Attendance: ArcFace embedding per track until the identity is confirmed
     by `confirm_votes` agreeing matches; then the track is not re-embedded
  4. Engagement: MediaPipe cues on an enlarged face crop, sliding window per
     track, classified every `window` samples
  5. Activity: YOLOv8 behaviour boxes; a face is linked to the box that
     contains its centre (highest confidence)
  6. Aggregator: per-student timeline and class-level summaries
"""
import time
from collections import Counter, defaultdict, deque

import cv2
import numpy as np

from src.attendance.embedders import ArcFaceEmbedder, Detector
from src.attendance.gallery import Gallery
from src.engagement.features import FaceCueExtractor
from src.engagement.models import TRI_NAMES, rule_based
from src.pipeline.tracker import IoUTracker


class ClassroomAnalytics:
    def __init__(self, gallery: Gallery, activity_weights=None, det_size=1280,
                 engagement_model=None, window=30, cue_fps=3.0, confirm_votes=3,
                 activity_every=1, engagement_every=1):
        self.det = Detector(det_size=det_size)
        self.arc = ArcFaceEmbedder(self.det)
        self.gallery = gallery
        self.cues = FaceCueExtractor()
        self.tracker = IoUTracker()
        self.activity = None
        if activity_weights:
            from ultralytics import YOLO
            self.activity = YOLO(str(activity_weights))
        self.eng_model = engagement_model  # callable(seqs (N,T,59)) -> labels, or None = rule-based
        self.window, self.cue_fps, self.confirm_votes = window, cue_fps, confirm_votes
        self.activity_every, self.engagement_every = activity_every, engagement_every
        self.votes = defaultdict(Counter)
        self.identity = {}
        self.cue_buf = defaultdict(lambda: deque(maxlen=window))
        self.timeline = []
        self.timing = defaultdict(list)
        self.frame_idx = 0

    def _crop(self, img, box, scale=1.6):
        x1, y1, x2, y2 = box
        cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale / 2
        h, w = img.shape[:2]
        a, b = int(max(0, cx - s)), int(max(0, cy - s))
        c, d = int(min(w, cx + s)), int(min(h, cy + s))
        return img[b:d, a:c]

    def process(self, frame_bgr, t_sec=None):
        T = {}
        t0 = time.perf_counter()
        faces = self.det.detect(frame_bgr)
        T["detect"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        tids = self.tracker.update([f.bbox for f in faces])
        T["track"] = time.perf_counter() - t0

        # attendance: only embed tracks that are not yet confirmed
        t0 = time.perf_counter()
        todo = [(f, tid) for f, tid in zip(faces, tids) if tid not in self.identity]
        if todo:
            E = self.arc.embed(frame_bgr, [f for f, _ in todo])
            preds, _ = self.gallery.identify(E)
            for (f, tid), p in zip(todo, preds):
                if p is not None:
                    self.votes[tid][p] += 1
                    who, n = self.votes[tid].most_common(1)[0]
                    if n >= self.confirm_votes and who not in self.identity.values():
                        self.identity[tid] = who
        T["attendance"] = time.perf_counter() - t0

        # engagement cues
        t0 = time.perf_counter()
        eng = {}
        if self.frame_idx % self.engagement_every == 0:
            for f, tid in zip(faces, tids):
                crop = self._crop(frame_bgr, f.bbox)
                if crop.size == 0:
                    continue
                v = self.cues.frame_vector(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                self.cue_buf[tid].append(v if v is not None else np.full(59, np.nan, np.float32))
                if len(self.cue_buf[tid]) == self.window:
                    seq = np.stack(self.cue_buf[tid])[None]
                    lab = (self.eng_model(seq) if self.eng_model else rule_based(seq, self.cue_fps))[0]
                    eng[tid] = TRI_NAMES[int(lab)]
        T["engagement"] = time.perf_counter() - t0

        # activity
        t0 = time.perf_counter()
        act = {}
        if self.activity is not None and self.frame_idx % self.activity_every == 0:
            r = self.activity.predict(frame_bgr, imgsz=640, verbose=False, conf=0.3)[0]
            boxes = r.boxes.xyxy.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for f, tid in zip(faces, tids):
                cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
                inside = [(c, k) for b, k, c in zip(boxes, cls, conf)
                          if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]]
                if inside:
                    act[tid] = r.names[max(inside)[1]]
        T["activity"] = time.perf_counter() - t0

        for f, tid in zip(faces, tids):
            self.timeline.append({"frame": self.frame_idx, "t": t_sec, "track": tid,
                                  "student": self.identity.get(tid), "engagement": eng.get(tid),
                                  "activity": act.get(tid), "bbox": [float(x) for x in f.bbox]})
        T["total"] = sum(T.values())
        for k, v in T.items():
            self.timing[k].append(v)
        self.frame_idx += 1
        return {"faces": len(faces), "identities": dict(self.identity), "engagement": eng, "activity": act}

    def attendance(self):
        return sorted(set(self.identity.values()))

    def summary(self):
        df_eng = Counter(r["engagement"] for r in self.timeline if r["engagement"])
        df_act = Counter(r["activity"] for r in self.timeline if r["activity"])
        return {"present": self.attendance(), "n_present": len(self.attendance()),
                "engagement_distribution": dict(df_eng), "activity_distribution": dict(df_act),
                "latency_ms_median": {k: 1e3 * float(np.median(v)) for k, v in self.timing.items()}}
