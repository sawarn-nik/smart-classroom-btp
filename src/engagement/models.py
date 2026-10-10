import numpy as np
import torch
from torch import nn

from src.engagement.features import clip_features

# DAiSEE engagement levels 0..3 -> 3 classes used by the system
TRI = {0: 0, 1: 0, 2: 1, 3: 2}
TRI_NAMES = ["disengaged", "partially engaged", "engaged"]

# column slices inside the 59-d frame vector
GROUPS = {"au": slice(0, 52), "pose": slice(52, 55), "gaze": slice(55, 59)}


def to_tri(y4):
    return np.vectorize(TRI.get)(np.asarray(y4))


def seq_to_frames(seq):
    return [None if np.isnan(f).any() else f for f in seq]


def clip_matrix(seqs, fps, groups=("au", "pose", "gaze")):
    """Clip descriptors restricted to selected feature groups."""
    keep = np.concatenate([np.arange(59)[GROUPS[g]] for g in groups])
    rows = []
    for s in seqs:
        v = clip_features(seq_to_frames(s), fps)
        if v is None:
            v = np.full(59 * 2 + 5, np.nan, np.float32)
        mean, std, extra = v[:59], v[59:118], v[118:]
        rows.append(np.concatenate([mean[keep], std[keep], extra]))
    return np.nan_to_num(np.stack(rows), nan=0.0)


def rule_based(seqs, fps, away_thr=0.5, closed_thr=0.3, partial_thr=0.2):
    """Heuristic baseline: mostly away / eyes closed -> disengaged; some -> partial."""
    preds = []
    for s in seqs:
        v = clip_features(seq_to_frames(s), fps)
        if v is None:
            preds.append(0); continue
        visible, away, closed = v[118], v[119], v[120]
        bad = max(away, closed, 1 - visible)
        preds.append(0 if (away > away_thr or closed > closed_thr) else 1 if bad > partial_thr else 2)
    return np.array(preds)


class GRUEngagement(nn.Module):
    """Small temporal model over per-frame cue vectors (+ face-present flag)."""

    def __init__(self, in_dim=60, hidden=64, n_classes=3, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4 * hidden, n_classes)

    def forward(self, x):
        h, _ = self.gru(self.norm(x))
        pooled = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(self.drop(pooled))


def seq_tensor(seqs):
    present = ~np.isnan(seqs).any(-1, keepdims=True)
    x = np.concatenate([np.nan_to_num(seqs, nan=0.0), present.astype(np.float32)], -1)
    return torch.from_numpy(x.astype(np.float32))
