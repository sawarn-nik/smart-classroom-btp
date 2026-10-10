"""Attendance experiments on cached LFW embeddings.

1. LFW verification (6000 standard pairs, 10-fold).
2. Closed/open-set identification with k-shot enrolment (k=1,3).
3. Robustness of identification under classroom-like corruptions.
Writes results/metrics/attendance_lfw.json and figures.
"""
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from src.attendance.metrics import open_set, verification_summary
from src.common.paths import FIGURES, LFW_PAIRS, METRICS, PROCESSED

MODELS = ["arcface", "facenet"]
CONDS = ["clean", "occlusion", "occlusion_eyes", "blur", "blur_strong", "low_light",
         "low_res", "low_res_20", "low_res_14"]
LABEL = {"arcface": "ArcFace (ResNet-50)", "facenet": "FaceNet (InceptionResNetV1)"}


def load(cond):
    d = np.load(PROCESSED / f"lfw_emb_{cond}.npz")
    return {k: d[k] for k in d.files}


def read_pairs():
    lines = LFW_PAIRS.read_text().strip().split("\n")[1:]
    pairs = []
    for ln in lines:
        p = ln.split("\t")
        if len(p) == 3:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg", 1))
        else:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg", 0))
    return pairs


def verification(clean):
    index = {p: i for i, p in enumerate(clean["paths"])}
    pairs = read_pairs()
    out, curves = {}, {}
    labels = np.array([l for _, _, l in pairs])
    for m in MODELS:
        E = clean[m]
        s = np.array([E[index[a]] @ E[index[b]] for a, b, _ in pairs])
        out[m] = verification_summary(s, labels)
        curves[m] = s
    n_missed = int(sum(not clean["found"][index[a]] or not clean["found"][index[b]] for a, b, _ in pairs))
    out["pairs_with_detection_failure"] = n_missed

    from sklearn.metrics import roc_curve
    plt.figure(figsize=(4.5, 4))
    for m in MODELS:
        fpr, tpr, _ = roc_curve(labels, curves[m])
        plt.plot(fpr, tpr, label=f"{LABEL[m]} (AUC={out[m]['auc']:.4f})")
    plt.xscale("log"); plt.xlim(1e-4, 1); plt.xlabel("False accept rate"); plt.ylabel("True accept rate")
    plt.grid(alpha=.3); plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(FIGURES / "attendance_lfw_roc.pdf"); plt.savefig(FIGURES / "attendance_lfw_roc.png", dpi=200)
    plt.close()
    return out


def split(data, k, ids):
    by = defaultdict(list)
    for i, n in enumerate(data["names"]):
        by[n].append(i)
    gal, probes = {}, []
    for n in ids:
        idx = by[n]
        gal[n] = idx[:k]
        probes += [(n, i) for i in idx[k:]]
    return gal, probes


def identification(gallery_data, probe_data, k, ids, impostor_names, max_probes_per_id=None):
    """Gallery templates always from clean images; probes from `probe_data`."""
    gal, probes = split(gallery_data, k, ids)
    p_index = {p: i for i, p in enumerate(probe_data["paths"])}
    if max_probes_per_id:
        cnt = defaultdict(int); kept = []
        for n, i in probes:
            if cnt[n] < max_probes_per_id:
                kept.append((n, i)); cnt[n] += 1
        probes = kept
    imp_idx = [i for i, n in enumerate(probe_data["names"]) if n in impostor_names]
    res = {}
    for m in MODELS:
        G = gallery_data[m]
        T = np.stack([G[gal[n]].mean(0) for n in ids])
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        P = probe_data[m]
        rows = [p_index[gallery_data["paths"][i]] for _, i in probes]
        S = P[rows] @ T.T
        found = probe_data["found"][rows]
        top = S.argmax(1)
        correct = (np.array([ids.index(n) for n, _ in probes]) == top) & found
        mated = np.where(found, S.max(1), -1)
        Simp = P[imp_idx] @ T.T
        nonmated = np.where(probe_data["found"][imp_idx], Simp.max(1), -1)
        tpir1, th1 = open_set(mated, correct, nonmated, 0.01)
        tpir5, th5 = open_set(mated, correct, nonmated, 0.05)
        res[m] = {"rank1": float(correct.mean()), "tpir@fpir=0.01": tpir1, "thr@fpir=0.01": th1,
                  "tpir@fpir=0.05": tpir5, "detect_rate": float(found.mean()),
                  "n_probes": len(probes), "n_impostors": len(imp_idx), "n_gallery": len(ids)}
    return res


def main():
    clean = load("clean")
    out = {"verification": verification(clean)}

    counts = defaultdict(int)
    for n in clean["names"]:
        counts[n] += 1
    ids5 = sorted(n for n, c in counts.items() if c >= 5)
    singles = {n for n, c in counts.items() if c == 1}

    out["identification"] = {}
    for k in (1, 3):
        out["identification"][f"k={k}"] = identification(clean, clean, k, ids5, singles)

    out["robustness"] = {}
    for cond in CONDS:
        try:
            probe = clean if cond == "clean" else load(cond)
        except FileNotFoundError:
            print("missing", cond); continue
        imp = set(probe["names"]) & singles
        out["robustness"][cond] = identification(clean, probe, 3, ids5, imp, max_probes_per_id=3)

    (METRICS / "attendance_lfw.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    if len(out["robustness"]) > 1:
        conds = list(out["robustness"])
        x = np.arange(len(conds)); w = 0.38
        plt.figure(figsize=(8, 3.4))
        for j, m in enumerate(MODELS):
            v = [100 * out["robustness"][c][m]["rank1"] for c in conds]
            b = plt.bar(x + (j - .5) * w, v, w, label=LABEL[m])
            plt.bar_label(b, fmt="%.1f", fontsize=7)
        plt.xticks(x, [c.replace("_", " ") for c in conds], rotation=25, fontsize=8); plt.ylabel("Rank-1 accuracy (%)")
        plt.ylim(0, 105); plt.legend(fontsize=8, loc="lower left"); plt.tight_layout()
        plt.savefig(FIGURES / "attendance_robustness.pdf"); plt.savefig(FIGURES / "attendance_robustness.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
