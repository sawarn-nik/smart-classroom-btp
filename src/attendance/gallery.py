"""Enrolment gallery and open-set matching used for marking attendance."""
import numpy as np

from src.attendance.embedders import l2norm


class Gallery:
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.ids = []
        self.templates = None

    def enroll(self, embeddings_by_id):
        """embeddings_by_id: {student_id: (k, d) array}. Template = mean of k shots."""
        self.ids = list(embeddings_by_id)
        self.templates = l2norm(np.stack([np.asarray(embeddings_by_id[i]).mean(0)
                                          for i in self.ids]))
        return self

    def scores(self, probes):
        return np.asarray(probes) @ self.templates.T

    def identify(self, probes):
        """Returns (pred_ids, best_scores); pred is None when below threshold."""
        s = self.scores(probes)
        best = s.argmax(1)
        top = s[np.arange(len(s)), best]
        preds = [self.ids[b] if t >= self.threshold else None for b, t in zip(best, top)]
        return preds, top

    def mark_attendance(self, probe_embeddings):
        """One-to-one assignment: a student can be matched by at most one face."""
        if len(probe_embeddings) == 0:
            return set()
        s = self.scores(probe_embeddings)
        present, used = set(), set()
        order = np.dstack(np.unravel_index(np.argsort(-s, axis=None), s.shape))[0]
        for fi, gi in order:
            if s[fi, gi] < self.threshold:
                break
            if fi in used or self.ids[gi] in present:
                continue
            used.add(fi)
            present.add(self.ids[gi])
        return present
