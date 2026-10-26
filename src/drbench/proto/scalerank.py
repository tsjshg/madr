"""ScaleRank prototype: rank preservation at user-chosen scales.

Objective (see docs/scalerank.md)::

    L(Y) = sum_i sum_K p(K) * (1 - A_K(i))

where A_K(i) is the probability that a true K-neighbour j of i is embedded
closer than a non-neighbour m, and p(K) is the scale profile.  It is
optimised stochastically: draw K ~ p, j from the top-K of i in the original
space, m from outside it, and apply the smooth triplet loss

    l(i, j, m) = softplus(log(1 + d_ij^2) - log(1 + d_im^2)).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

_CHUNK_ELEMS = 4_000_000


def scale_profile(n: int, gamma: float = 1.0, k0: float | None = None, width: float = 1.0) -> np.ndarray:
    """p(K) for K = 1..n-2 (index 0 of the result is K=1).

    Power law K^-gamma by default; with ``k0`` a log-normal band around k0
    of log-width ``width``.
    """
    k = np.arange(1, n - 1, dtype=np.float64)
    if k0 is None:
        p = k ** (-gamma)
    else:
        p = np.exp(-((np.log(k) - np.log(k0)) ** 2) / (2 * width**2)) / k
    return p / p.sum()


def neighbour_order(X: np.ndarray, local_scaling: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rank tables of the original space.

    order[i, r-1] is the point with rank r among the neighbours of i, and
    rank[i, j] is the rank of j (0 for j == i).

    With ``local_scaling=k`` distances are divided by sqrt(sigma_i sigma_j),
    sigma = distance to the k-th neighbour (Zelnik-Manor & Perona 2004).  This
    reduces hubness: anti-hubs, which otherwise sit in nobody's neighbour list,
    get ranked close to their own neighbours, much as t-SNE's per-point
    bandwidth does.
    """
    n = X.shape[0]
    sigma = None
    if local_scaling:
        dist, _ = NearestNeighbors(n_neighbors=local_scaling + 1).fit(X).kneighbors(X)
        sigma = np.maximum(dist[:, -1], 1e-12)
    order = np.empty((n, n - 1), dtype=np.int32)
    rank = np.zeros((n, n), dtype=np.int32)
    step = max(1, _CHUNK_ELEMS // n)
    for start in range(0, n, step):
        rows = np.arange(start, min(start + step, n))
        D = cdist(X[rows], X)
        if sigma is not None:
            D /= np.sqrt(sigma[rows, None] * sigma[None, :])
        D[np.arange(len(rows)), rows] = -np.inf
        o = np.argsort(D, axis=1, kind="stable")
        order[rows] = o[:, 1:]
        # write through a slice: rank[rows] with an index array would be a copy
        np.put_along_axis(rank[start : start + len(rows)], o, np.arange(n, dtype=np.int32)[None, :], axis=1)
    return order, rank


def sample_triplets(
    order: np.ndarray,
    p: np.ndarray,
    per_point: int,
    rng: np.random.Generator,
    rank: np.ndarray | None = None,
    low_nbrs: np.ndarray | None = None,
    hard: float = 0.0,
    Y: np.ndarray | None = None,
    hard_pos: int = 1,
    hard_c: float | None = None,
):
    """Anchors i, in-neighbours j (rank <= K) and out-neighbours m (rank > K).

    With probability ``hard`` the out-neighbour is drawn from the current
    low-dimensional neighbours of i instead (positions < 2K of ``low_nbrs``):
    those intruders are what lowers R_NX(K), whereas a uniform out-neighbour
    is usually already far away.  Draws that turn out to be true K-neighbours,
    or need K beyond the cached low-dim neighbour list, fall back to uniform.

    With ``hard_pos > 1`` (and ``Y``), the in-neighbour is the one among
    ``hard_pos`` candidates from the top-K that is currently farthest from i
    in ``Y``: the extrusion counterpart of the hard negatives, which otherwise
    reward a point for drifting away from everyone.

    Returns ``(i, j, m, w, is_hard)``.  ``w`` is 1 except for hard negatives when
    ``hard_c`` is set: an intruder of original rank r gets
    ``min(1, (r - K) / (hard_c * K))``, the trustworthiness weighting, so
    near-misses that no 2-D layout can avoid are pushed only weakly.
    """
    n = order.shape[0]
    i = np.repeat(np.arange(n), per_point)
    K = rng.choice(np.arange(1, n - 1), size=i.size, p=p)
    if hard_pos > 1 and Y is not None:
        cand = order[i[:, None], (rng.random((i.size, hard_pos)) * K[:, None]).astype(np.int64)]
        d2 = ((Y[cand] - Y[i][:, None, :]) ** 2).sum(axis=2)
        j = cand[np.arange(i.size), np.argmax(d2, axis=1)]
    else:
        j = order[i, (rng.random(i.size) * K).astype(np.int64)]
    m = order[i, K + (rng.random(i.size) * (n - 1 - K)).astype(np.int64)]
    if hard > 0 and low_nbrs is not None:
        width = np.minimum(2 * K, low_nbrs.shape[1])
        cand = low_nbrs[i, (rng.random(i.size) * width).astype(np.int64)]
        use = (rng.random(i.size) < hard) & (2 * K <= low_nbrs.shape[1]) & (rank[i, cand] > K)
        m = np.where(use, cand, m)
        w = np.ones(i.size)
        if hard_c is not None:
            w[use] = np.minimum(1.0, (rank[i[use], m[use]] - K[use]) / (hard_c * K[use]))
        return i, j, m, w, use
    return i, j, m, np.ones(i.size), np.zeros(i.size, dtype=bool)


def triplet_loss_and_grad(Y: np.ndarray, i, j, m, w=None, freeze_m=None) -> tuple[float, np.ndarray]:
    """Mean (optionally weighted) triplet loss and its gradient w.r.t. Y (summed per point).

    ``freeze_m`` marks triplets whose out-neighbour m is held fixed (stop-gradient):
    only the anchor moves away from it.
    """
    dij = Y[i] - Y[j]
    dim = Y[i] - Y[m]
    sij = (dij**2).sum(axis=1)
    sim = (dim**2).sum(axis=1)
    z = np.log1p(sij) - np.log1p(sim)
    w = np.ones(z.size) if w is None else w
    loss = w * np.logaddexp(0.0, z)
    s = w / (1.0 + np.exp(-z))  # d loss / d z
    gij = (2 * s / (1 + sij))[:, None] * dij  # d loss / d y_i through d_ij
    gim = (-2 * s / (1 + sim))[:, None] * dim  # d loss / d y_i through d_im
    n, dim_out = Y.shape
    G = np.zeros_like(Y)
    for c in range(dim_out):
        G[:, c] = (
            np.bincount(i, weights=gij[:, c] + gim[:, c], minlength=n)
            - np.bincount(j, weights=gij[:, c], minlength=n)
            - np.bincount(m, weights=gim[:, c] if freeze_m is None else np.where(freeze_m, 0.0, gim[:, c]), minlength=n)
        )
    return float(loss.mean()), G


class ScaleRank:
    def __init__(
        self,
        gamma: float = 1.0,
        k0: float | None = None,
        width: float = 1.0,
        n_components: int = 2,
        n_epochs: int = 800,
        per_point: int = 32,
        lr: float = 1.0,
        init: str = "pca",
        init_scale: float = 1.0,
        hard: float = 0.3,
        hard_k: int = 128,
        refresh_every: int = 5,
        hard_pos: int = 1,
        hard_decay: bool = False,
        hard_c: float | None = None,
        hard_one_sided: bool = False,
        local_scaling: int | None = None,
        local_scaling_kmax: int | None = None,
        optimizer: str = "adam",
        snapshot_every: int = 0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.gamma, self.k0, self.width = gamma, k0, width
        self.n_components = n_components
        self.n_epochs, self.per_point, self.lr = n_epochs, per_point, lr
        self.init, self.init_scale = init, init_scale
        self.hard, self.hard_k, self.refresh_every = hard, hard_k, refresh_every
        if optimizer not in ("adam", "adam_global", "sgd"):
            raise ValueError(f"unknown optimizer {optimizer!r}")
        self.optimizer = optimizer
        self.hard_pos, self.hard_decay, self.hard_c = hard_pos, hard_decay, hard_c
        self.hard_one_sided = hard_one_sided
        self.local_scaling, self.local_scaling_kmax = local_scaling, local_scaling_kmax
        self.snapshot_every = snapshot_every
        self.random_state = random_state
        self.verbose = verbose
        self.loss_history_: list[float] = []

    def _init(self, X, rng):
        if self.init == "pca":
            Y = PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        else:
            Y = rng.normal(size=(X.shape[0], self.n_components))
        return Y / Y[:, 0].std() * self.init_scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        p = scale_profile(n, self.gamma, self.k0, self.width)
        # Local scaling either everywhere, or only for K <= local_scaling_kmax:
        # small neighbourhoods then use hubness-reduced ranks while larger
        # scales keep the raw ranks (global structure).  Triplets are split
        # between the two tables in proportion to the profile's mass.
        tables = []
        if self.local_scaling and self.local_scaling_kmax:
            cut = min(self.local_scaling_kmax, n - 2)
            small = np.r_[p[:cut], np.zeros(n - 2 - cut)]
            n_small = int(round(self.per_point * small.sum()))
            if n_small > 0:
                tables.append((*neighbour_order(X, self.local_scaling), small / small.sum(), n_small))
            if self.per_point - n_small > 0 and small.sum() < 1:
                large = p - small
                tables.append((*neighbour_order(X), large / large.sum(), self.per_point - n_small))
        else:
            tables.append((*neighbour_order(X, self.local_scaling), p, self.per_point))
        hard_k = min(self.hard_k, n - 1)
        Y = self._init(X, rng)

        # Linearly decaying step size.  "adam" normalises every coordinate of
        # every point separately; "adam_global" shares one second-moment
        # estimate across all points, so a point with weak, noisy gradients
        # takes small steps; "sgd" is momentum SGD on the raw gradient.
        b1, b2, eps = 0.9, 0.999, 1e-8
        mom = np.zeros_like(Y)
        low_nbrs = None
        vel = np.zeros_like(Y) if self.optimizer == "adam" else 0.0
        self.loss_history_ = []
        self.snapshots_: list[tuple[int, np.ndarray]] = [(0, Y.copy())] if self.snapshot_every else []
        for t in range(1, self.n_epochs + 1):
            if self.hard > 0 and (t - 1) % self.refresh_every == 0:
                low_nbrs = NearestNeighbors(n_neighbors=hard_k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
            hard = self.hard * (1 - (t - 1) / self.n_epochs) if self.hard_decay else self.hard
            parts = [
                sample_triplets(order, pk, cnt, rng, rank, low_nbrs, hard, Y, self.hard_pos, self.hard_c)
                for order, rank, pk, cnt in tables
            ]
            i, j, m, w, is_hard = (np.concatenate(a) for a in zip(*parts))
            loss, G = triplet_loss_and_grad(Y, i, j, m, w, is_hard if self.hard_one_sided else None)
            G /= self.per_point
            step = self.lr * (1 - (t - 1) / self.n_epochs)
            mom = b1 * mom + (1 - b1) * G
            if self.optimizer == "sgd":
                Y -= step * mom
            else:
                g2 = G**2 if self.optimizer == "adam" else float(np.mean(G**2))
                vel = b2 * vel + (1 - b2) * g2
                Y -= step * (mom / (1 - b1**t)) / (np.sqrt(vel / (1 - b2**t)) + eps)
            self.loss_history_.append(loss)
            if self.snapshot_every and t % self.snapshot_every == 0:
                self.snapshots_.append((t, Y.copy()))
            if self.verbose and (t % 50 == 0 or t == 1):
                print(f"epoch {t:4d}  loss {loss:.4f}")
        self.embedding_ = Y
        return Y
