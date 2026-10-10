"""Evaluate trained activity detectors on the held-out SCB test split.

Because the SCB subsets are partially labelled, each class is scored only on
test images from the subset(s) that annotate it ("source-aware" AP). The naive
all-images numbers are reported too, for transparency.
Also measures inference latency (batch 1, 640 px) on CPU and MPS.
"""
import argparse
import json
import time

import numpy as np
from ultralytics import YOLO

from src.common.paths import CKPT, METRICS, PROCESSED

SCB = PROCESSED / "scb_merged"
SOURCE_CLASSES = {"hrw": [0, 1, 2], "bowturn": [3, 4], "discuss": [5]}


def per_class(metrics):
    """{class_idx: (P, R, AP50, AP50-95)} from an ultralytics DetMetrics."""
    box = metrics.box
    out = {}
    for j, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(j)
        out[int(c)] = tuple(float(x) for x in (p, r, ap50, ap))
    return out


def latency(model, device, n=50):
    imgs = sorted((SCB / "images" / "test").glob("*.jpg"))[:n]
    model.predict(str(imgs[0]), device=device, imgsz=640, verbose=False)  # warm-up
    ts = []
    for p in imgs:
        t0 = time.perf_counter()
        model.predict(str(p), device=device, imgsz=640, verbose=False)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names under checkpoints/activity")
    args = ap.parse_args()
    names = ["hand_raising", "reading", "writing", "bowing_head", "turning_head", "discussing"]
    out = {}
    for run in args.runs:
        w = CKPT / "activity" / run / "weights" / "best.pt"
        model = YOLO(str(w))
        res = {"weights": str(w), "params_M": sum(p.numel() for p in model.model.parameters()) / 1e6}
        naive = model.val(data=str(SCB / "data.yaml"), split="test", imgsz=640, batch=16, device="mps",
                          plots=False, verbose=False, project=str(CKPT / "activity" / "_eval"), name=run, exist_ok=True)
        res["naive_all_images"] = {"mAP50": float(naive.box.map50), "mAP50-95": float(naive.box.map)}
        aware = {}
        for src, cls in SOURCE_CLASSES.items():
            m = model.val(data=str(SCB / f"data_test_{src}.yaml"), split="test", imgsz=640, batch=16,
                          device="mps", plots=(src == "hrw"), verbose=False,
                          project=str(CKPT / "activity" / "_eval"), name=f"{run}_{src}", exist_ok=True)
            pc = per_class(m)
            for c in cls:
                if c in pc:
                    aware[names[c]] = dict(zip(["P", "R", "AP50", "AP50-95"], pc[c]))
        res["source_aware_per_class"] = aware
        res["source_aware_mAP50"] = float(np.mean([v["AP50"] for v in aware.values()]))
        res["source_aware_mAP50-95"] = float(np.mean([v["AP50-95"] for v in aware.values()]))
        res["latency_ms_cpu"] = latency(model, "cpu")
        res["latency_ms_mps"] = latency(model, "mps")
        out[run] = res
        print(run, json.dumps(res, indent=2))
    path = METRICS / "activity_scb.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(prev | out, indent=2))


if __name__ == "__main__":
    main()
