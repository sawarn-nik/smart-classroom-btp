"""End-to-end attendance on synthetic multi-face classroom frames.

Each session: 30 enrolled students (3 clean LFW photos each), a random
70-95% of them present, plus 3 unenrolled visitors. Head crops from
held-out LFW photos are pasted on a 1920x1080 canvas in 5 rows, shrinking
from ~130 px (front) to ~38 px (back) to mimic perspective. Session-level
lighting and per-face blur are applied randomly. The whole frame then goes
through detection -> embedding -> one-to-one gallery assignment.
"""
import argparse
import json
import time
from collections import defaultdict

import cv2
import numpy as np

from src.attendance.degradations import low_light
from src.attendance.embedders import ArcFaceEmbedder, Detector, FaceNetEmbedder, largest_face
from src.attendance.gallery import Gallery
from src.attendance.metrics import attendance_prf
from src.common.paths import FIGURES, LFW_DIR, METRICS, PROCESSED

W, H = 1920, 1080
ROWS = 5
FACE_PX = np.linspace(130, 38, ROWS)  # head crop width per row, front -> back


def head_crop(path):
    img = cv2.imread(str(path))
    return img[45:205, 55:195]  # 160x140 centred on the funneled face


def build_frame(people, rng):
    canvas = np.full((H, W, 3), rng.integers(60, 140), np.uint8)
    canvas = cv2.add(canvas, rng.normal(0, 6, canvas.shape).astype(np.int8).astype(np.uint8))
    per_row = int(np.ceil(len(people) / ROWS))
    rng.shuffle(people)
    boxes = []
    for i, (pid, path) in enumerate(people):
        r, c = divmod(i, per_row)
        r = ROWS - 1 - r  # fill from the back row so front rows are the sparse ones
        size = int(FACE_PX[r] * rng.uniform(0.9, 1.1))
        crop = cv2.resize(head_crop(path), (size, int(size * 160 / 140)))
        if rng.random() < 0.25:
            crop = cv2.GaussianBlur(crop, (5, 5), 0)
        # back rows sit higher in the frame
        y = int(np.clip(H * (0.15 + 0.17 * (ROWS - 1 - r)), 0, H - crop.shape[0]))
        span = W / per_row
        x = int(np.clip(c * span + rng.uniform(0.1, 0.4) * span, 0, W - crop.shape[1]))
        canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        boxes.append((pid, r))
    if rng.random() < 0.3:
        canvas = low_light(canvas, gamma=1.8, noise=6, seed=int(rng.integers(1e6)))
    return canvas, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--class-size", type=int, default=30)
    ap.add_argument("--det-size", type=int, default=1280)
    ap.add_argument("--thr-arcface", type=float, default=None)
    ap.add_argument("--thr-facenet", type=float, default=None)
    ap.add_argument("--save-examples", type=int, default=2)
    args = ap.parse_args()

    # thresholds: operating point at FPIR=1% from the LFW open-set experiment
    thr = {"arcface": args.thr_arcface, "facenet": args.thr_facenet}
    lfw = METRICS / "attendance_lfw.json"
    if lfw.exists():
        j = json.loads(lfw.read_text())["identification"]["k=3"]
        for m in thr:
            thr[m] = thr[m] if thr[m] is not None else j[m]["thr@fpir=0.01"]
    thr = {m: (v if v is not None else {"arcface": 0.35, "facenet": 0.55}[m]) for m, v in thr.items()}
    print("thresholds", thr)

    clean = np.load(PROCESSED / "lfw_emb_clean.npz")
    names, paths = clean["names"], clean["paths"]
    by = defaultdict(list)
    for i, n in enumerate(names):
        by[n].append(i)
    ids5 = sorted(n for n, v in by.items() if len(v) >= 5)
    singles = sorted(n for n, v in by.items() if len(v) == 1)

    det = Detector(det_size=args.det_size)
    emb = {"arcface": ArcFaceEmbedder(det), "facenet": FaceNetEmbedder()}
    rng = np.random.default_rng(42)
    rows = []
    per_row_hits = {m: np.zeros(ROWS) for m in emb}
    per_row_tot = np.zeros(ROWS)
    lat = defaultdict(list)

    for s in range(args.sessions):
        roster = list(rng.choice(ids5, args.class_size, replace=False))
        galleries = {m: Gallery(thr[m]).enroll({n: clean[m][by[n][:3]] for n in roster}) for m in emb}
        present = [n for n in roster if rng.random() < rng.uniform(0.7, 0.95)]
        visitors = list(rng.choice(singles, 3, replace=False))
        people = [(n, LFW_DIR / paths[by[n][int(rng.integers(3, len(by[n])))]]) for n in present]
        people += [(f"visitor:{v}", LFW_DIR / paths[by[v][0]]) for v in visitors]
        frame, placed = build_frame(people, rng)
        if s < args.save_examples:
            cv2.imwrite(str(FIGURES / f"classroom_sim_example_{s}.jpg"), frame)

        t0 = time.perf_counter(); faces = det.detect(frame); t_det = time.perf_counter() - t0
        lat["detect"].append(t_det)
        rec = {"session": s, "n_present": len(present), "n_faces_placed": len(people),
               "n_detected": len(faces)}
        for m, e in emb.items():
            t0 = time.perf_counter(); E = e.embed(frame, faces); t_emb = time.perf_counter() - t0
            lat[m].append(t_emb)
            marked = galleries[m].mark_attendance(E)
            p, r, f1 = attendance_prf(marked, present)
            rec.update({f"{m}_precision": p, f"{m}_recall": r, f"{m}_f1": f1,
                        f"{m}_exact_roll": float(set(marked) == set(present))})
            for pid, row in placed:
                if not pid.startswith("visitor:"):
                    per_row_hits[m][row] += pid in marked
        for pid, row in placed:
            if not pid.startswith("visitor:"):
                per_row_tot[row] += 1
        rows.append(rec)
        print(s, {k: round(v, 3) if isinstance(v, float) else v for k, v in rec.items()})

    summary = {"config": vars(args) | {"thresholds": thr}, "sessions": rows}
    agg = {"detection_rate": float(np.sum([r["n_detected"] for r in rows]) / np.sum([r["n_faces_placed"] for r in rows]))}
    for m in emb:
        for k in ("precision", "recall", "f1", "exact_roll"):
            agg[f"{m}_{k}"] = float(np.mean([r[f"{m}_{k}"] for r in rows]))
        agg[f"{m}_recall_by_row_front_to_back"] = (per_row_hits[m] / np.maximum(per_row_tot, 1)).tolist()
        agg[f"{m}_embed_ms_per_frame"] = 1e3 * float(np.mean(lat[m]))
    agg["detect_ms_per_frame"] = 1e3 * float(np.mean(lat["detect"]))
    agg["face_px_by_row_front_to_back"] = FACE_PX.tolist()
    summary["aggregate"] = agg
    (METRICS / f"attendance_classroom_sim_det{args.det_size}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
