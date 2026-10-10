"""Merge the three SCB subsets into one 6-class YOLO dataset.

Classes: 0 hand_raising, 1 reading, 2 writing, 3 bowing_head, 4 turning_head, 5 discussing
- Identical images across subsets (same md5) are merged and labels unioned.
- An image that is in `val` in any subset goes to our held-out `test` split
  (prevents leakage); 10% of the remaining images (by hash) form `val`,
  used only for checkpoint selection.
- Sources are recorded so evaluation can restrict each class to images whose
  source subset actually annotates that class (the subsets are partially labelled).
"""
import hashlib
import json
import shutil
from collections import Counter, defaultdict

from src.common.paths import PROCESSED, RAW

SCB = RAW / "scb"
SOURCES = {
    "hrw": (SCB / "SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17", {0: 0, 1: 1, 2: 2}),
    "bowturn": (SCB / "SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17", {0: 3, 1: 4}),
    "discuss": (SCB / "SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17", {0: 5}),
}
NAMES = ["hand_raising", "reading", "writing", "bowing_head", "turning_head", "discussing"]
SOURCE_CLASSES = {s: sorted(m.values()) for s, (_, m) in SOURCES.items()}
OUT = PROCESSED / "scb_merged"


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    images = {}  # md5 -> {"path", "sources", "splits", "labels"}
    for src, (root, cmap) in SOURCES.items():
        for split in ("train", "val"):
            for img in sorted((root / "images" / split).glob("*.jpg")):
                h = md5(img)
                rec = images.setdefault(h, {"path": img, "sources": set(), "splits": set(), "labels": []})
                rec["sources"].add(src); rec["splits"].add(split)
                lbl = root / "labels" / split / (img.stem + ".txt")
                if lbl.exists():
                    for ln in lbl.read_text().split("\n"):
                        p = ln.split()
                        if len(p) == 5:
                            rec["labels"].append(f"{cmap[int(p[0])]} {' '.join(p[1:])}")

    manifest = defaultdict(list)
    stats = {"train": Counter(), "val": Counter(), "test": Counter()}
    n_img = Counter()
    for h, rec in images.items():
        if "val" in rec["splits"]:
            split = "test"
        else:
            split = "val" if int(h[:8], 16) % 10 == 0 else "train"
        for sub in ("images", "labels"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
        dst = OUT / "images" / split / f"{h}.jpg"
        dst.symlink_to(rec["path"])
        labels = sorted(set(rec["labels"]))
        (OUT / "labels" / split / f"{h}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        manifest[split].append({"id": h, "sources": sorted(rec["sources"])})
        n_img[split] += 1
        stats[split].update(NAMES[int(l.split()[0])] for l in labels)

    # per-source test lists for partial-label-aware evaluation
    for src in SOURCES:
        lst = [str(OUT / "images" / "test" / f"{m['id']}.jpg") for m in manifest["test"] if src in m["sources"]]
        (OUT / f"test_{src}.txt").write_text("\n".join(lst) + "\n")

    yaml = (f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (OUT / "data.yaml").write_text(yaml)
    for src in SOURCES:
        (OUT / f"data_test_{src}.yaml").write_text(yaml.replace("test: images/test", f"test: test_{src}.txt"))

    summary = {"n_images": dict(n_img), "boxes_per_class": {k: dict(v) for k, v in stats.items()},
               "source_classes": {s: [NAMES[c] for c in cs] for s, cs in SOURCE_CLASSES.items()},
               "n_unique_images": len(images),
               "n_multi_source_images": sum(len(r["sources"]) > 1 for r in images.values()),
               "n_split_conflicts_moved_to_test": sum(r["splits"] == {"train", "val"} for r in images.values())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "manifest.json").write_text(json.dumps(manifest))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
