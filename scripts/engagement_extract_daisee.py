"""Extract per-frame face cues from DAiSEE clips (official Train/Validation/Test split).

Expected layout (as shipped by IIT-H):
  data/raw/DAiSEE/DataSet/{Train,Validation,Test}/<subject>/<clipid>/<clipid>.avi|.mp4
  data/raw/DAiSEE/Labels/{TrainLabels,ValidationLabels,TestLabels}.csv
Output: data/processed/daisee_{split}.npz with seq (N, T, 59; NaN = no face),
labels (N, 4: boredom, engagement, confusion, frustration), clip ids, subjects.
"""
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.paths import PROCESSED, RAW
from src.engagement.features import FaceCueExtractor

ROOT = RAW / "DAiSEE"
SPLITS = {"Train": "TrainLabels", "Validation": "ValidationLabels", "Test": "TestLabels"}


def sample_frames(path, fps_out, max_frames):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps / fps_out)))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="debug: clips per split")
    args = ap.parse_args()

    clips = {p.stem: p for ext in ("*.avi", "*.mp4") for p in (ROOT / "DataSet").rglob(ext)}
    fx = FaceCueExtractor()
    for split, lblname in SPLITS.items():
        out = PROCESSED / f"daisee_{split.lower()}.npz"
        if out.exists():
            print("skip", out); continue
        df = pd.read_csv(ROOT / "Labels" / f"{lblname}.csv")
        df.columns = [c.strip() for c in df.columns]
        if args.limit:
            df = df.head(args.limit)
        seqs, labels, ids, subjects, missing = [], [], [], [], 0
        for _, r in tqdm(df.iterrows(), total=len(df), desc=split, mininterval=30):
            cid = str(r["ClipID"]).rsplit(".", 1)[0]
            path = clips.get(cid)
            if path is None:
                missing += 1; continue
            frames = sample_frames(path, args.fps, args.max_frames)
            seq = np.full((args.max_frames, 59), np.nan, np.float32)
            for t, fr in enumerate(frames):
                v = fx.frame_vector(fr)
                if v is not None:
                    seq[t] = v
            seqs.append(seq)
            labels.append([r["Boredom"], r["Engagement"], r["Confusion"], r["Frustration"]])
            ids.append(cid); subjects.append(path.parent.parent.name)
        np.savez_compressed(out, seq=np.stack(seqs), labels=np.array(labels), ids=np.array(ids),
                            subjects=np.array(subjects), feature_names=np.array(fx.feature_names()),
                            fps=args.fps)
        print(split, "clips", len(seqs), "missing videos", missing)


if __name__ == "__main__":
    main()
