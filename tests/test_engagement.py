import numpy as np
import pytest
import torch

from src.engagement.features import clip_features, eye_gaze, rotation_to_euler, L_EYE
from src.engagement.models import GRUEngagement, clip_matrix, rule_based, seq_tensor, to_tri


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def test_euler_recovers_single_axis_rotations():
    assert np.allclose(rotation_to_euler(rot_y(30)), [30, 0, 0], atol=1e-6)
    assert np.allclose(rotation_to_euler(rot_x(-20)), [0, -20, 0], atol=1e-6)


def test_eye_gaze_centre_is_zero():
    lm = np.zeros((478, 3))
    lm[L_EYE["outer"]] = [0, 0.5, 0]; lm[L_EYE["inner"]] = [1, 0.5, 0]
    lm[L_EYE["top"]] = [0.5, 0, 0]; lm[L_EYE["bottom"]] = [0.5, 1, 0]
    lm[L_EYE["iris"]] = [0.5, 0.5, 0]
    assert np.allclose(eye_gaze(lm, L_EYE), (0, 0), atol=1e-4)
    lm[L_EYE["iris"]] = [1, 0.5, 0]
    assert np.allclose(eye_gaze(lm, L_EYE), (1, 0), atol=1e-4)


def frame(yaw=0, pitch=0, blink=0.0):
    v = np.zeros(59, np.float32)
    v[9] = v[10] = blink
    v[52], v[53] = yaw, pitch
    return v


def test_clip_features_behaviour_stats():
    frames = [frame()] * 5 + [frame(yaw=40)] * 3 + [None] * 2
    v = clip_features(frames, fps_sampled=3)
    visible, away, closed = v[118], v[119], v[120]
    assert v.shape == (123,)
    assert visible == pytest.approx(0.8) and away == pytest.approx(3 / 8) and closed == 0
    assert clip_features([None, None], 3) is None


def test_blink_count():
    frames = [frame(blink=b) for b in (0, .9, 0, .9, 0, 0)]
    v = clip_features(frames, fps_sampled=6)  # 1 second of video, 2 blinks
    assert v[121] == pytest.approx(120.0)


def test_rule_based_and_matrix():
    attentive = np.stack([frame() for _ in range(10)])
    away = np.stack([frame(yaw=60) for _ in range(10)])
    seqs = np.stack([attentive, away])
    assert list(rule_based(seqs, 3)) == [2, 0]
    X = clip_matrix(seqs, 3, groups=("pose",))
    assert X.shape == (2, 3 * 2 + 5)


def test_tri_mapping():
    assert list(to_tri([0, 1, 2, 3])) == [0, 0, 1, 2]


def test_gru_shapes_with_missing_frames():
    seqs = np.random.default_rng(0).normal(size=(4, 30, 59)).astype(np.float32)
    seqs[0, 5:9] = np.nan
    x = seq_tensor(seqs)
    assert x.shape == (4, 30, 60) and not torch.isnan(x).any()
    assert GRUEngagement()(x).shape == (4, 3)
