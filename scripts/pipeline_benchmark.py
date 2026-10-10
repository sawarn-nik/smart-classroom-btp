"""End-to-end latency of the unified pipeline on real classroom photos (SCB test split).

Frames are independent photos, so tracking never confirms identities and
every face is embedded on every frame: this is the worst case for attendance.
"""
import argparse
import json

import cv2
import numpy as np

from src.attendance.gallery import Gallery
from src.common.paths import CKPT, FIGURES, METRICS, PROCESSED
from src.pipeline.system import ClassroomAnalytics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--activity-run", default="yolov8n_scb_e30")
    ap.add_argument("--det-size", type=int, default=640)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    gallery = Gallery(0.4).enroll({f"s{i}": rng.normal(size=(1, 512)) for i in range(60)})
    weights = CKPT / "activity" / args.activity_run / "weights" / "best.pt"
    sys_ = ClassroomAnalytics(gallery, activity_weights=weights if weights.exists() else None,
                              det_size=args.det_size)
    imgs = sorted((PROCESSED / "scb_merged" / "images" / "test").glob("*.jpg"))
    imgs = [imgs[i] for i in rng.choice(len(imgs), args.frames, replace=False)]
    n_faces, sizes = [], []
    for k, p in enumerate(imgs):
        fr = cv2.imread(str(p))
        sizes.append(fr.shape[:2])
        out = sys_.process(fr)
        n_faces.append(out["faces"])
        if k == 0:
            sys_.timing.clear()  # drop warm-up frame
        if k < 3:
            vis = fr.copy()
            for r in sys_.timeline:
                if r["frame"] == sys_.frame_idx - 1:
                    x1, y1, x2, y2 = map(int, r["bbox"])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    lab = r["activity"] or ""
                    cv2.putText(vis, lab, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(str(FIGURES / f"pipeline_example_{k}.jpg"), vis)
    T = sys_.timing
    res = {"frames": len(imgs) - 1, "det_size": args.det_size,
           "faces_per_frame_mean": float(np.mean(n_faces)), "faces_per_frame_max": int(np.max(n_faces)),
           "median_frame_hw": [int(x) for x in np.median(np.array(sizes), 0)],
           "latency_ms": {k: {"median": 1e3 * float(np.median(v)), "mean": 1e3 * float(np.mean(v)),
                              "p90": 1e3 * float(np.percentile(v, 90))} for k, v in T.items()},
           "activity_enabled": sys_.activity is not None}
    res["fps_median"] = 1e3 / res["latency_ms"]["total"]["median"]
    # per-face cost of the face-level modules
    nf = np.array(n_faces[1:]).clip(1)
    for k in ("attendance", "engagement"):
        res[f"{k}_ms_per_face"] = 1e3 * float(np.median(np.array(T[k]) / nf))
    (METRICS / f"pipeline_latency_det{args.det_size}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
