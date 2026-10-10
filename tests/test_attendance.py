import numpy as np

from src.attendance.degradations import CONDITIONS
from src.attendance.embedders import l2norm
from src.attendance.gallery import Gallery
from src.attendance.metrics import attendance_prf, lfw_10fold_accuracy, open_set


def make_ids(n=5, d=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = l2norm(rng.normal(size=(n, d)))
    return {f"s{i}": l2norm(centres[i] + 0.05 * rng.normal(size=(3, d))) for i in range(n)}, centres


def test_gallery_identifies_and_rejects_unknown():
    shots, centres = make_ids()
    g = Gallery(threshold=0.5).enroll(shots)
    preds, _ = g.identify(centres)
    assert preds == list(shots)
    stranger = l2norm(np.random.default_rng(9).normal(size=(1, 32)))
    assert g.identify(stranger)[0] == [None]


def test_mark_attendance_one_to_one():
    shots, centres = make_ids()
    g = Gallery(threshold=0.5).enroll(shots)
    # two faces of the same student must not mark two people
    probes = np.stack([centres[0], centres[0], centres[2]])
    assert g.mark_attendance(probes) == {"s0", "s2"}
    assert g.mark_attendance(np.zeros((0, 32))) == set()


def test_attendance_prf():
    assert attendance_prf({"a", "b"}, {"a", "b"}) == (1.0, 1.0, 1.0)
    p, r, f = attendance_prf({"a", "x"}, {"a", "b"})
    assert (p, r) == (0.5, 0.5) and abs(f - 0.5) < 1e-9


def test_lfw_10fold_perfect_separation():
    s = np.r_[np.full(300, 0.8), np.full(300, 0.1)]
    y = np.r_[np.ones(300), np.zeros(300)]
    order = np.random.default_rng(0).permutation(600)
    acc, std, th = lfw_10fold_accuracy(s[order], y[order])
    assert acc == 1.0 and 0.1 <= th < 0.8


def test_open_set_threshold():
    nonmated = np.linspace(0, 0.5, 1000)
    tpir, t = open_set([0.9, 0.9, 0.3], [True, False, True], nonmated, fpir=0.01)
    assert abs(t - 0.495) < 0.01 and abs(tpir - 1 / 3) < 1e-9


def test_degradations_keep_shape_and_dtype():
    img = np.random.default_rng(0).integers(0, 255, (250, 250, 3), dtype=np.uint8)
    for name, fn in CONDITIONS.items():
        out = fn(img)
        assert out.shape == img.shape and out.dtype == np.uint8, name
    assert CONDITIONS["low_light"](img).mean() < img.mean()


def test_center_face_prefers_centred_over_larger():
    from src.attendance.embedders import Face, center_face
    big_corner = Face(np.array([0, 0, 120, 120.]), None, 0.9)
    small_centre = Face(np.array([100, 100, 150, 150.]), None, 0.9)
    assert center_face([big_corner, small_centre], (250, 250, 3)) == [small_centre]
    assert center_face([], (250, 250, 3)) == []
