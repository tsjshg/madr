import numpy as np
import pytest
from scipy.spatial.distance import cdist

from drbench.proto.scalerank import (
    ScaleRank,
    neighbour_order,
    sample_triplets,
    scale_profile,
    triplet_loss_and_grad,
)


def test_scale_profile():
    p = scale_profile(100, gamma=1.0)
    assert p.shape == (98,) and p.sum() == pytest.approx(1.0)
    assert np.allclose(p[0] / p[9], 10.0)  # 1/K weighting
    assert np.allclose(scale_profile(100, gamma=0.0), 1 / 98)
    band = scale_profile(1000, k0=30, width=0.5)
    assert 15 < np.argmax(band * np.arange(1, 999)) + 1 < 45  # mass in log K peaks near k0


def test_triplets_respect_ranks():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    order, rank = neighbour_order(X)
    for a in range(80):
        assert (rank[a, order[a]] == np.arange(1, 80)).all()
    p = scale_profile(80, gamma=0.5)
    low_nbrs = np.argsort(cdist(X[:, :2], X[:, :2]), axis=1)[:, 1:31]  # any low-dim layout
    for hard in (0.0, 1.0):
        i, j, m = sample_triplets(order, p, 50, rng, rank, low_nbrs, hard)
        assert (j != i).all() and (m != i).all()
        assert (rank[i, j] < rank[i, m]).all()


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(12, 2))
    i, j, m = rng.integers(0, 12, (3, 40))
    keep = (i != j) & (i != m) & (j != m)
    i, j, m = i[keep], j[keep], m[keep]
    _, G = triplet_loss_and_grad(Y, i, j, m)
    eps = 1e-6
    num = np.zeros_like(Y)
    for a in range(12):
        for c in range(2):
            Yp, Ym = Y.copy(), Y.copy()
            Yp[a, c] += eps
            Ym[a, c] -= eps
            num[a, c] = (triplet_loss_and_grad(Yp, i, j, m)[0] - triplet_loss_and_grad(Ym, i, j, m)[0]) / (2 * eps)
    assert np.allclose(G / len(i), num, atol=1e-6)


def test_recovers_a_2d_layout():
    from drbench.metrics import rank_metrics

    X = np.random.default_rng(2).uniform(size=(300, 2))
    Y = ScaleRank(gamma=1.0, init="random", n_epochs=300, random_state=0).fit_transform(X)
    assert rank_metrics(X, Y).r_nx[10] > 0.8
