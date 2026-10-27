"""Quality metrics for low-dimensional embeddings.

Rank-based metrics follow Lee & Verleysen (2009) "Quality assessment of
dimensionality reduction: Rank-based criteria" and Lee et al. (2015) for the
log-K weighted AUC of R_NX.  Ranks are computed row by row in chunks so the
N x N co-ranking matrix is never materialised: every quantity we need can be
accumulated from per-row rank vectors.

Above ``EXACT_MAX_N`` points :func:`evaluate` uses only a random subset of
rows (anchors).  Every rank metric is a mean over rows, so the subset gives
an unbiased estimate at O(anchors x N) cost; per-point overlap@K is then
computed for all points from kNN lists instead.

Conventions
-----------
- rank_h[i, j] / rank_l[i, j]: rank of j among the neighbours of i in the
  high / low space, 1..N-1 (self gets 0 and is excluded from all sums).
  Ties are broken by index (stable sort), as in the legacy CoRankingQ code.
- K is a neighbourhood size, 1 <= K <= N-2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

DEFAULT_KS = (5, 10, 30, 100)
_CHUNK_ELEMS = 4_000_000  # rows * N per chunk; bounds memory to ~100 MB
EXACT_MAX_N = 10_000  # above this, evaluate() samples anchor rows
N_ANCHORS = 2_000


def _row_ranks(D: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Ranks along axis 1 of a distance block; the diagonal element gets 0."""
    D = D.copy()
    D[np.arange(len(rows)), rows] = -np.inf
    order = np.argsort(D, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(ranks, order, np.arange(D.shape[1], dtype=np.int32)[None, :], axis=1)
    return ranks


def iter_rank_chunks(X: np.ndarray, Y: np.ndarray, chunk_rows: int | None = None, anchors=None):
    """Yield (rows, rank_h, rank_l) blocks covering all rows (or just ``anchors``)."""
    n = X.shape[0]
    anchors = np.arange(n) if anchors is None else np.asarray(anchors)
    if chunk_rows is None:
        chunk_rows = max(1, _CHUNK_ELEMS // n)
    for start in range(0, len(anchors), chunk_rows):
        rows = anchors[start : start + chunk_rows]
        rh = _row_ranks(cdist(X[rows], X), rows)
        rl = _row_ranks(cdist(Y[rows], Y), rows)
        yield rows, rh, rl


@dataclass
class RankMetrics:
    n: int
    q_nx: np.ndarray  # index K -> Q_NX(K), K = 1..N-2 (index 0 unused)
    r_nx: np.ndarray  # index K -> R_NX(K)
    auc_r_nx: float
    trustworthiness: dict[int, float]
    continuity: dict[int, float]
    qscore_legacy: float  # sum_ij |rank_h - rank_l| / N  (the 2022 "Qスコア")
    footrule: float  # mean |rank_h - rank_l| / (N-1); ~1/3 for a random layout
    pointwise_overlap: dict[int, np.ndarray] = field(repr=False)  # NaN off the anchors
    n_anchors: int = 0  # rows the estimates are based on (N when exact)


def rank_metrics(
    X: np.ndarray, Y: np.ndarray, ks=DEFAULT_KS, chunk_rows: int | None = None, anchors=None
) -> RankMetrics:
    """All co-ranking based metrics in a single pass over the data.

    With ``anchors`` (row indices) only those rows of the co-ranking
    matrix are used and every metric is the corresponding row-mean estimate.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"inconsistent sizes: high={n} low={Y.shape[0]}")
    ks = [k for k in ks if 1 <= k < n / 2]
    s = n if anchors is None else len(anchors)

    max_rank_counts = np.zeros(n, dtype=np.int64)
    trust_pen = {k: 0.0 for k in ks}
    cont_pen = {k: 0.0 for k in ks}
    overlap = {k: np.zeros(n) if anchors is None else np.full(n, np.nan) for k in ks}
    abs_diff_sum = 0.0

    for rows, rh, rl in iter_rank_chunks(X, Y, chunk_rows, anchors):
        m = np.maximum(rh, rl)
        max_rank_counts += np.bincount(m.ravel(), minlength=n)
        abs_diff_sum += np.abs(rh.astype(np.int64) - rl).sum()
        for k in ks:
            in_h = (rh >= 1) & (rh <= k)
            in_l = (rl >= 1) & (rl <= k)
            # intrusions: close in the embedding, far in the original space
            trust_pen[k] += np.where(in_l & ~in_h, rh - k, 0).sum()
            # extrusions: close in the original space, far in the embedding
            cont_pen[k] += np.where(in_h & ~in_l, rl - k, 0).sum()
            overlap[k][rows] = (in_h & in_l).sum(axis=1) / k

    # Q_NX(K) = |{(i,j): rank_h <= K and rank_l <= K}| / (K N); the self pair
    # has max rank 0 and is dropped by starting the cumulative sum at index 1.
    # With anchors, N rows become s rows.
    cum = np.cumsum(max_rank_counts[1:])  # cum[K-1] = #pairs with max rank <= K
    ks_all = np.arange(1, n - 1)
    q_nx = np.zeros(n - 1)
    q_nx[1:] = cum[: n - 2] / (ks_all * s)
    r_nx = np.zeros(n - 1)
    r_nx[1:] = ((n - 1) * q_nx[1:] - ks_all) / (n - 1 - ks_all)
    auc = float(np.sum(r_nx[1:] / ks_all) / np.sum(1.0 / ks_all))

    def _norm(k):
        return 2.0 / (s * k * (2 * n - 3 * k - 1))

    return RankMetrics(
        n=n,
        q_nx=q_nx,
        r_nx=r_nx,
        auc_r_nx=auc,
        trustworthiness={k: 1.0 - _norm(k) * trust_pen[k] for k in ks},
        continuity={k: 1.0 - _norm(k) * cont_pen[k] for k in ks},
        qscore_legacy=abs_diff_sum / s,
        footrule=abs_diff_sum / (s * (n - 1) * (n - 1)),
        pointwise_overlap=overlap,
        n_anchors=s,
    )


def knn_overlap(X: np.ndarray, Y: np.ndarray, ks=DEFAULT_KS) -> dict[int, np.ndarray]:
    """Per-point |kNN_X(i) & kNN_Y(i)| / k for every point, from kNN lists.

    Same quantity as ``RankMetrics.pointwise_overlap`` (up to distance ties)
    without the O(N^2) rank tables.
    """
    kmax = max(ks)
    nx = NearestNeighbors(n_neighbors=kmax + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    ny = NearestNeighbors(n_neighbors=kmax + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    out = {}
    for k in ks:
        a, b = np.sort(nx[:, :k], axis=1), np.sort(ny[:, :k], axis=1)
        # count common ids per row: merge two sorted rows via searchsorted on offset ids
        off = (np.arange(len(X)) * len(X))[:, None]
        pos = np.searchsorted((b + off).ravel(), (a + off).ravel())
        pos = np.minimum(pos, b.size - 1)
        hit = (b + off).ravel()[pos] == (a + off).ravel()
        out[k] = hit.reshape(a.shape).sum(axis=1) / k
    return out


def distance_spearman(
    X: np.ndarray, Y: np.ndarray, max_pairs: int = 200_000, seed: int = 0
) -> float:
    """Spearman correlation of pairwise distances (global structure).

    Uses a random subset of pairs when N(N-1)/2 exceeds ``max_pairs``.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dh = np.linalg.norm(X[i] - X[j], axis=1)
    dl = np.linalg.norm(Y[i] - Y[j], axis=1)
    return float(spearmanr(dh, dl).statistic)


def local_radius(Z: np.ndarray, k: int = 10) -> np.ndarray:
    """Mean distance to the k nearest neighbours (a local density proxy)."""
    dist, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return dist[:, 1:].mean(axis=1)


def density_correlation(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Spearman correlation of log local radii, as used to evaluate densMAP."""
    rx = np.log(local_radius(X, k) + 1e-12)
    ry = np.log(local_radius(Y, k) + 1e-12)
    return float(spearmanr(rx, ry).statistic)


def knn_accuracy(Z: np.ndarray, labels: np.ndarray, k: int = 10, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated k-NN classification accuracy of the labels in space Z."""
    _, counts = np.unique(labels, return_counts=True)
    folds = int(min(folds, counts.min()))
    if folds < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    clf = KNeighborsClassifier(n_neighbors=k)
    return float(cross_val_score(clf, Z, labels, cv=cv).mean())


def evaluate(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray | None = None,
    ks=DEFAULT_KS,
) -> tuple[dict, dict]:
    """Compute every metric.

    Returns ``(scalars, arrays)``: a flat dict of scalar metrics for the
    results table, and a dict of arrays (R_NX curve, per-point overlap).
    Above ``EXACT_MAX_N`` points the rank metrics are estimated from
    ``N_ANCHORS`` random rows (fixed seed) and recorded as ``n_anchors``.
    """
    n = X.shape[0]
    anchors = None
    if n > EXACT_MAX_N:
        anchors = np.sort(np.random.default_rng(0).choice(n, N_ANCHORS, replace=False))
    rm = rank_metrics(X, Y, ks=ks, anchors=anchors)
    scalars: dict[str, float] = {
        "auc_rnx": rm.auc_r_nx,
        "qscore_legacy": rm.qscore_legacy,
        "footrule": rm.footrule,
        "dist_spearman": distance_spearman(X, Y),
        "density_corr": density_correlation(X, Y),
    }
    for k, v in rm.trustworthiness.items():
        scalars[f"trust@{k}"] = v
    for k, v in rm.continuity.items():
        scalars[f"cont@{k}"] = v
    for k in ks:
        if 1 <= k < rm.n - 1:
            scalars[f"rnx@{k}"] = float(rm.r_nx[k])
    if labels is not None and labels.dtype.kind in "iuUSO":
        scalars["knn_acc"] = knn_accuracy(Y, labels)
    if anchors is not None:
        scalars["n_anchors"] = len(anchors)
    arrays = {"r_nx": rm.r_nx}
    overlap = rm.pointwise_overlap if anchors is None else knn_overlap(X, Y, [k for k in ks if k < n / 2])
    for k, v in overlap.items():
        arrays[f"overlap@{k}"] = v
    return scalars, arrays
