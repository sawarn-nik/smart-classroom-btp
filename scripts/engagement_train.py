"""Train/evaluate engagement classifiers on extracted DAiSEE cues.

Models: rule-based heuristic, logistic regression, gradient boosting (clip
descriptors), and a BiGRU over the frame sequence. Hyper-parameters / epochs
are chosen on Validation; all reported numbers are on the official Test split.
Also runs a feature-group ablation (AU / pose / gaze) with gradient boosting.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.common.paths import FIGURES, METRICS, PROCESSED
from src.engagement.models import (TRI_NAMES, GRUEngagement, clip_matrix, rule_based, seq_tensor,
                                   to_tri)

SEED = 0


def load(split):
    d = np.load(PROCESSED / f"daisee_{split}.npz")
    return d["seq"], d["labels"][:, 1].astype(int), float(d["fps"])


def scores(y, p):
    return {"acc": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "weighted_f1": float(f1_score(y, p, average="weighted"))}


def train_gru(Xtr, ytr, Xva, yva, n_classes, epochs=40, lr=1e-3, device="cpu"):
    torch.manual_seed(SEED)
    model = GRUEngagement(in_dim=Xtr.shape[-1], n_classes=n_classes).to(device)
    w = torch.tensor(len(ytr) / (n_classes * np.bincount(ytr, minlength=n_classes).clip(1)), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    ytr_t = torch.from_numpy(ytr)
    best, best_state, history = -1, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(model(Xtr[b].to(device)), ytr_t[b].to(device))
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(yva, pv, average="macro")
        history.append({"epoch": ep, "loss": float(loss), "val_macro_f1": float(f1)})
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def main():
    (s_tr, ytr4, fps), (sva, yva4, _), (ste, yte4, _) = load("train"), load("validation"), load("test")
    out = {"n": {"train": len(ytr4), "val": len(yva4), "test": len(yte4)},
           "label_dist_4": {s: np.bincount(y, minlength=4).tolist() for s, y in
                            [("train", ytr4), ("val", yva4), ("test", yte4)]},
           "face_found_ratio_test": float((~np.isnan(ste).any(-1)).mean())}
    results, preds = {}, {}

    for task, (ytr, yva, yte, nc) in {"3class": (to_tri(ytr4), to_tri(yva4), to_tri(yte4), 3),
                                     "4class": (ytr4, yva4, yte4, 4)}.items():
        res = {}
        Xtr, Xva, Xte = (clip_matrix(s, fps) for s in (s_tr, sva, ste))
        sw = compute_sample_weight("balanced", ytr)

        res["majority"] = scores(yte, np.full_like(yte, np.bincount(ytr).argmax()))
        if task == "3class":
            # thresholds tuned on validation
            best = max(((a, c, p) for a in (.3, .4, .5, .6) for c in (.2, .3, .4) for p in (.1, .2, .3)),
                       key=lambda t: f1_score(yva, rule_based(sva, fps, *t), average="macro"))
            preds["rule"] = rule_based(ste, fps, *best)
            res["rule_based"] = scores(yte, preds["rule"]) | {"thresholds": best}

        best_c = max((0.01, 0.1, 1.0), key=lambda C: f1_score(
            yva, make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
            .fit(Xtr, ytr).predict(Xva), average="macro"))
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced")).fit(Xtr, ytr)
        res["logreg"] = scores(yte, lr.predict(Xte)) | {"C": best_c}

        def gb(depth, lr_):
            return HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr_, max_iter=300,
                                                  random_state=SEED)
        cfgs = [(d, l) for d in (3, 5) for l in (0.05, 0.1)]
        best_cfg = max(cfgs, key=lambda c: f1_score(yva, gb(*c).fit(Xtr, ytr, sample_weight=sw).predict(Xva), average="macro"))
        gbm = gb(*best_cfg).fit(Xtr, ytr, sample_weight=sw)
        preds[f"gbm_{task}"] = gbm.predict(Xte)
        res["gbm"] = scores(yte, preds[f"gbm_{task}"]) | {"cfg": best_cfg}

        Ttr, Tva, Tte = seq_tensor(s_tr), seq_tensor(sva), seq_tensor(ste)
        model, hist = train_gru(Ttr, ytr, Tva, yva, nc)
        with torch.no_grad():
            preds[f"gru_{task}"] = model.eval()(Tte).argmax(1).numpy()
        res["bigru"] = scores(yte, preds[f"gru_{task}"]) | {"best_val_epoch": int(np.argmax([h["val_macro_f1"] for h in hist]))}
        torch.save(model.state_dict(), PROCESSED / f"engagement_bigru_{task}.pt")

        if task == "3class":
            abl = {}
            for groups in [("au",), ("pose",), ("gaze",), ("pose", "gaze"), ("au", "pose"), ("au", "pose", "gaze")]:
                A = [clip_matrix(s, fps, groups) for s in (s_tr, ste)]
                m = gb(*best_cfg).fit(A[0], ytr, sample_weight=sw)
                abl["+".join(groups)] = scores(yte, m.predict(A[1]))
            res["ablation_gbm"] = abl
            best_name = max(("gbm", "bigru"), key=lambda k: res[k]["macro_f1"])
            p = preds[f"{'gbm' if best_name == 'gbm' else 'gru'}_3class"]
            cm = confusion_matrix(yte, p, labels=[0, 1, 2])
            res["confusion_matrix_best"] = {"model": best_name, "matrix": cm.tolist()}
            plt.figure(figsize=(3.8, 3.3))
            cmn = cm / cm.sum(1, keepdims=True).clip(1)
            plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center", fontsize=7,
                             color="white" if cmn[i, j] > .5 else "black")
            short = ["Dis.", "Partial", "Eng."]
            plt.xticks(range(3), short); plt.yticks(range(3), short)
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
            plt.savefig(FIGURES / "engagement_confusion.pdf"); plt.savefig(FIGURES / "engagement_confusion.png", dpi=200)
            plt.close()
        results[task] = res

    out["results"] = results
    out["class_names_3"] = TRI_NAMES
    (METRICS / "engagement_daisee.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
