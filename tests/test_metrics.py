import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.manifold import trustworthiness

from drbench.metrics import evaluate, rank_metrics


def brute_coranking(X, Y):
    """Direct port of the legacy CoRankingQ2 (stable argsort, self at rank 0)."""
    n = X.shape[0]
    hdm, ldm = cdist(X, X), cdist(Y, Y)
    np.fill_diagonal(hdm, -np.inf)
    np.fill_diagonal(ldm, -np.inf)
    Q = np.zeros((n, n), dtype=int)
    for i in range(n):
        rh = np.empty(n, int)
        rl = np.empty(n, int)
        rh[np.argsort(hdm[i], kind="stable")] = np.arange(n)
        rl[np.argsort(ldm[i], kind="stable")] = np.arange(n)
        for j in range(n):
            Q[rh[j], rl[j]] += 1
    return Q


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    Y = X[:, :2] + 0.3 * rng.normal(size=(60, 2))
    return X, Y


def test_identity_is_perfect():
    X = np.random.default_rng(1).normal(size=(50, 2))
    rm = rank_metrics(X, X)
    assert np.allclose(rm.r_nx[1:], 1.0)
    assert rm.footrule == 0.0
    assert all(v == pytest.approx(1.0) for v in rm.trustworthiness.values())
    assert all(v == pytest.approx(1.0) for v in rm.continuity.values())


def test_q_nx_matches_brute_force(data):
    X, Y = data
    n = X.shape[0]
    Q = brute_coranking(X, Y)
    rm = rank_metrics(X, Y)
    for k in range(1, n - 1):
        expected = Q[1 : k + 1, 1 : k + 1].sum() / (k * n)
        assert rm.q_nx[k] == pytest.approx(expected)


def test_legacy_qscore_matches(data):
    X, Y = data
    n = X.shape[0]
    Q = brute_coranking(X, Y)
    idx = np.arange(n)
    legacy = (Q * np.abs(idx[:, None] - idx[None, :])).sum() / n
    assert rank_metrics(X, Y).qscore_legacy == pytest.approx(legacy)


@pytest.mark.parametrize("k", [3, 5, 10])
def test_trustworthiness_matches_sklearn(data, k):
    X, Y = data
    rm = rank_metrics(X, Y, ks=(k,))
    assert rm.trustworthiness[k] == pytest.approx(trustworthiness(X, Y, n_neighbors=k))
    # continuity is trustworthiness with the roles of the spaces swapped
    assert rm.continuity[k] == pytest.approx(trustworthiness(Y, X, n_neighbors=k))


def test_chunking_does_not_change_results(data):
    X, Y = data
    a = rank_metrics(X, Y, chunk_rows=7)
    b = rank_metrics(X, Y)
    assert np.allclose(a.r_nx, b.r_nx)
    assert a.qscore_legacy == b.qscore_legacy
    assert a.trustworthiness == b.trustworthiness


def test_random_layout_is_chance_level():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 10))
    Y = rng.normal(size=(400, 2))
    rm = rank_metrics(X, Y)
    assert abs(rm.auc_r_nx) < 0.05
    assert rm.footrule == pytest.approx(1 / 3, abs=0.02)


def test_evaluate_outputs(data):
    X, Y = data
    labels = (X[:, 0] > 0).astype(int)
    scalars, arrays = evaluate(X, Y, labels)
    assert {"auc_rnx", "trust@10", "knn_acc", "density_corr", "dist_spearman"} <= scalars.keys()
    assert arrays["overlap@10"].shape == (60,)
