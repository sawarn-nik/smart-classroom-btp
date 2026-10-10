import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def lfw_10fold_accuracy(scores, labels, folds=10):
    """Standard LFW protocol: threshold picked on 9 folds, accuracy on the 10th."""
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    idx = np.array_split(np.arange(len(scores)), folds)
    cand = np.linspace(-1, 1, 801)
    accs, ths = [], []
    for k in range(folds):
        test = idx[k]
        train = np.concatenate([idx[j] for j in range(folds) if j != k])
        tr_acc = [((scores[train] >= t) == labels[train]).mean() for t in cand]
        t = cand[int(np.argmax(tr_acc))]
        accs.append(((scores[test] >= t) == labels[test]).mean())
        ths.append(t)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(ths))


def tar_at_far(scores, labels, far):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(far, fpr, tpr))


def verification_summary(scores, labels):
    acc, std, th = lfw_10fold_accuracy(scores, labels)
    return {"acc": acc, "acc_std": std, "threshold": th,
            "auc": float(roc_auc_score(labels, scores)),
            "tar@far=1e-2": tar_at_far(scores, labels, 1e-2),
            "tar@far=1e-3": tar_at_far(scores, labels, 1e-3)}


def open_set(mated_scores, mated_correct, nonmated_scores, fpir=0.01):
    """TPIR at a given FPIR (NIST FRVT style).

    mated_scores: top-1 score of probes whose identity is enrolled
    mated_correct: whether that top-1 was the right identity
    nonmated_scores: top-1 score of probes whose identity is not enrolled
    """
    nonmated_scores = np.asarray(nonmated_scores)
    t = float(np.quantile(nonmated_scores, 1 - fpir))
    hit = (np.asarray(mated_scores) >= t) & np.asarray(mated_correct)
    return float(hit.mean()), t


def attendance_prf(predicted_present, true_present):
    p, t = set(predicted_present), set(true_present)
    tp = len(p & t)
    prec = tp / len(p) if p else 1.0
    rec = tp / len(t) if t else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1
