"""Steady-state behaviour on a static classroom shot (camera fixed, students seated).

The same SCB classroom photo is fed as a 40-frame clip with small jitter/noise.
Faces in frame 0 are enrolled as the roster, so after `confirm_votes` frames
identities are cached and attendance stops costing anything. Also reports how
often MediaPipe returns cues for face crops of different sizes.
"""
import json

import cv2
import numpy as np

from src.attendance.gallery import Gallery
from src.common.paths import CKPT, METRICS, PROCESSED
from src.pipeline.system import ClassroomAnalytics


def main(n_frames=40, n_images=5):
    rng = np.random.default_rng(1)
    imgs = sorted((PROCESSED / "scb_merged" / "images" / "test").glob("*.jpg"))
    weights = CKPT / "activity" / "yolov8n_scb_e30" / "weights" / "best.pt"
    runs, cue_ok = [], {"<40px": [0, 0], "40-80px": [0, 0], ">80px": [0, 0]}
    picked = 0
    for p in imgs[::7]:
        base = cv2.imread(str(p))
        sys_ = ClassroomAnalytics(Gallery(0.3), activity_weights=weights, det_size=640)
        faces = sys_.det.detect(base)
        if not 8 <= len(faces) <= 40:
            continue
        E = sys_.arc.embed(base, faces)
        sys_.gallery.enroll({f"s{i}": E[i:i + 1] for i in range(len(faces))})
        for t in range(n_frames):
            M = np.float32([[1, 0, rng.normal(0, 1)], [0, 1, rng.normal(0, 1)]])
            fr = cv2.warpAffine(base, M, base.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
            fr = np.clip(fr + rng.normal(0, 3, fr.shape), 0, 255).astype(np.uint8)
            sys_.process(fr, t_sec=t / 15)
        # cue success by face size (on the clean frame)
        for f in faces:
            w = f.bbox[2] - f.bbox[0]
            key = "<40px" if w < 40 else "40-80px" if w < 80 else ">80px"
            crop = sys_._crop(base, f.bbox)
            ok = sys_.cues.frame_vector(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) is not None
            cue_ok[key][0] += ok; cue_ok[key][1] += 1
        T = {k: np.array(v) * 1e3 for k, v in sys_.timing.items()}
        runs.append({"image": p.name, "faces": len(faces), "identified": len(sys_.attendance()),
                     "first5_total_ms": float(T["total"][:5].mean()),
                     "steady_total_ms": float(np.median(T["total"][10:])),
                     "steady_breakdown_ms": {k: float(np.median(v[10:])) for k, v in T.items()}})
        print(runs[-1])
        picked += 1
        if picked == n_images:
            break
    out = {"runs": runs,
           "mean_steady_total_ms": float(np.mean([r["steady_total_ms"] for r in runs])),
           "mean_first5_total_ms": float(np.mean([r["first5_total_ms"] for r in runs])),
           "identified_ratio": float(np.sum([r["identified"] for r in runs]) / np.sum([r["faces"] for r in runs])),
           "mediapipe_cue_success_by_face_width": {k: {"ok": a, "n": b, "rate": a / b if b else None}
                                                   for k, (a, b) in cue_ok.items()}}
    (METRICS / "pipeline_steady_state.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
