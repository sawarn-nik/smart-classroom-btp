"""Detect + embed every LFW image with both models; cache to data/processed.

Usage: python -m scripts.attendance_embed_lfw [--conditions clean occlusion ...]
For non-clean conditions only the identification subset (ids with >=5 images,
plus 1500 single-image impostors) is embedded to keep runtime down.
"""
import argparse
import json
import time

import cv2
import numpy as np
from tqdm import tqdm

from src.attendance.degradations import CONDITIONS
from src.attendance.embedders import ArcFaceEmbedder, Detector, FaceNetEmbedder, largest_face
from src.common.paths import LFW_DIR, PROCESSED


def list_images():
    people = sorted(p for p in LFW_DIR.iterdir() if p.is_dir())
    items = []
    for p in people:
        for f in sorted(p.glob("*.jpg")):
            items.append((p.name, f))
    return items


def subset_for_degradation(items, seed=0):
    from collections import Counter
    cnt = Counter(n for n, _ in items)
    ident = [it for it in items if cnt[it[0]] >= 5]
    singles = [it for it in items if cnt[it[0]] == 1]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(singles), 1500, replace=False)
    return ident + [singles[i] for i in sorted(pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    args = ap.parse_args()

    det = Detector(det_size=320)
    arc = ArcFaceEmbedder(det)
    fnet = FaceNetEmbedder()
    all_items = list_images()

    for cond in args.conditions:
        out = PROCESSED / f"lfw_emb_{cond}.npz"
        if out.exists():
            print("skip", out)
            continue
        items = all_items if cond == "clean" else subset_for_degradation(all_items)
        fn = CONDITIONS[cond]
        names, paths, ea, ef, found = [], [], [], [], []
        t_det, t_arc, t_fn = [], [], []
        for name, path in tqdm(items, desc=cond, mininterval=10):
            img = fn(cv2.imread(str(path)))
            t0 = time.perf_counter(); faces = largest_face(det.detect(img)); t1 = time.perf_counter()
            if faces:
                a = arc.embed(img, faces)[0]; t2 = time.perf_counter()
                f = fnet.embed(img, faces)[0]; t3 = time.perf_counter()
                t_det.append(t1 - t0); t_arc.append(t2 - t1); t_fn.append(t3 - t2)
            else:
                a = f = np.zeros(512, np.float32)
            names.append(name); paths.append(str(path.relative_to(LFW_DIR)))
            ea.append(a); ef.append(f); found.append(bool(faces))
        np.savez_compressed(out, names=np.array(names), paths=np.array(paths),
                            arcface=np.stack(ea), facenet=np.stack(ef), found=np.array(found))
        timing = {"n": len(items), "detect_rate": float(np.mean(found)),
                  "det_ms": 1e3 * float(np.mean(t_det)), "arcface_ms": 1e3 * float(np.mean(t_arc)),
                  "facenet_ms": 1e3 * float(np.mean(t_fn))}
        (PROCESSED / f"lfw_emb_{cond}_timing.json").write_text(json.dumps(timing, indent=2))
        print(cond, timing)


if __name__ == "__main__":
    main()
