"""Validate head-pose cues on real classroom photos using SCB behaviour labels.

For every face SCRFD finds in an SCB test image, the face is linked to the
labelled behaviour box that contains its centre. We compare
(a) MediaPipe availability (does Face Landmarker return anything?) and
(b) InsightFace 3D-68 head pose (pitch, yaw), available for every detected face,
across behaviour classes. If pose is meaningful, bowing_head should have the
lowest pitch and turning_head the largest |yaw|.
"""
import json
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_auc_score

from src.common.paths import FIGURES, INSIGHTFACE_ROOT, METRICS, PROCESSED
from src.engagement.features import FaceCueExtractor

SCB = PROCESSED / "scb_merged"
NAMES = ["hand_raising", "reading", "writing", "bowing_head", "turning_head", "discussing"]


def load_boxes(img_path, w, h):
    lbl = SCB / "labels" / "test" / (img_path.stem + ".txt")
    out = []
    for ln in lbl.read_text().split("\n"):
        p = ln.split()
        if len(p) == 5:
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
            out.append((c, (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def main(max_images=600):
    app = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_ROOT),
                       allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    mp = FaceCueExtractor()
    imgs = sorted((SCB / "images" / "test").glob("*.jpg"))
    rng = np.random.default_rng(0)
    imgs = [imgs[i] for i in rng.choice(len(imgs), min(max_images, len(imgs)), replace=False)]
    rows = []
    for p in imgs:
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        boxes = load_boxes(p, w, h)
        if not boxes:
            continue
        for f in app.get(im):
            cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
            inside = [b for b in boxes if b[1] <= cx <= b[3] and b[2] <= cy <= b[4]]
            if not inside:
                continue
            c = min(inside, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]))[0]  # tightest box
            x1, y1, x2, y2 = f.bbox
            s = max(x2 - x1, y2 - y1) * 1.1
            crop = im[int(max(0, cy - s)):int(cy + s), int(max(0, cx - s)):int(cx + s)]
            mp_ok = crop.size > 0 and mp.frame_vector(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) is not None
            pitch, yaw, roll = (float(v) for v in f.pose)
            rows.append({"cls": NAMES[c], "pitch": pitch, "yaw": yaw, "width": float(x2 - x1), "mp_ok": bool(mp_ok)})

    by = defaultdict(list)
    for r in rows:
        by[r["cls"]].append(r)
    summary = {}
    for c in NAMES:
        R = by[c]
        if not R:
            continue
        summary[c] = {"n_faces": len(R),
                      "pitch_median": float(np.median([r["pitch"] for r in R])),
                      "abs_yaw_median": float(np.median([abs(r["yaw"]) for r in R])),
                      "face_width_median": float(np.median([r["width"] for r in R])),
                      "mediapipe_success": float(np.mean([r["mp_ok"] for r in R]))}
    # separability: AUC of a single pose angle for the two head-behaviour classes vs the rest
    y_bow = np.array([r["cls"] == "bowing_head" for r in rows])
    y_turn = np.array([r["cls"] == "turning_head" for r in rows])
    out = {"n_images": len(imgs), "n_faces_linked": len(rows), "per_class": summary,
           "auc_bowing_vs_rest_from_-pitch": float(roc_auc_score(y_bow, [-r["pitch"] for r in rows])),
           "auc_turning_vs_rest_from_|yaw|": float(roc_auc_score(y_turn, [abs(r["yaw"]) for r in rows])),
           "mediapipe_success_overall": float(np.mean([r["mp_ok"] for r in rows])),
           "insightface_pose_success_overall": 1.0}
    (METRICS / "engagement_pose_validation_scb.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    order = [c for c in NAMES if c in summary]
    ax[0].boxplot([[r["pitch"] for r in by[c]] for c in order], showfliers=False)
    ax[0].set_ylabel("Pitch (deg)"); ax[0].set_xticks(range(1, len(order) + 1), [c.replace("_", "\n") for c in order], fontsize=7)
    ax[1].boxplot([[abs(r["yaw"]) for r in by[c]] for c in order], showfliers=False)
    ax[1].set_ylabel("|Yaw| (deg)"); ax[1].set_xticks(range(1, len(order) + 1), [c.replace("_", "\n") for c in order], fontsize=7)
    for a in ax:
        a.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIGURES / "engagement_pose_by_behaviour.pdf"); plt.savefig(FIGURES / "engagement_pose_by_behaviour.png", dpi=200)


if __name__ == "__main__":
    main()
