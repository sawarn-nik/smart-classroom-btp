"""Runs the full DAiSEE training script on a tiny synthetic dataset."""
import json

import numpy as np

import scripts.engagement_train as et


def fake_split(path, n, rng):
    y = rng.integers(0, 4, n)
    seq = rng.normal(size=(n, 30, 59)).astype(np.float32) * 0.1
    seq[:, :, 52] += (3 - y)[:, None] * 12  # more yaw for lower engagement
    seq[: n // 10, :5] = np.nan
    labels = np.stack([np.zeros(n), y, np.zeros(n), np.zeros(n)], 1).astype(int)
    np.savez_compressed(path, seq=seq, labels=labels, ids=np.arange(n).astype(str),
                        subjects=np.zeros(n).astype(str), feature_names=np.array([]), fps=3.0)


def test_engagement_train_runs(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for s, n in (("train", 160), ("validation", 60), ("test", 60)):
        fake_split(tmp_path / f"daisee_{s}.npz", n, rng)
    for mod_attr in ("PROCESSED", "METRICS", "FIGURES"):
        monkeypatch.setattr(et, mod_attr, tmp_path)
    monkeypatch.setattr(et, "train_gru", _short)
    et.main()
    out = json.loads((tmp_path / "engagement_daisee.json").read_text())
    r = out["results"]["3class"]
    assert {"majority", "rule_based", "logreg", "gbm", "bigru", "ablation_gbm"} <= set(r)
    assert r["gbm"]["macro_f1"] > r["majority"]["macro_f1"]
    assert (tmp_path / "engagement_confusion.pdf").exists()


_orig = et.train_gru


def _short(*a, **k):
    return _orig(*a, epochs=3, **k)
