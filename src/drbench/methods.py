"""Embedding methods under a common signature ``fn(X, seed, **params) -> Y``.

Methods are addressed by a spec string, ``name`` or ``name:key=val,key=val``
(e.g. ``tsne:perplexity=5``), so parameter variants need no extra code.
A new method (MADR or otherwise) only has to be decorated with
``@register("name")``.
"""

from __future__ import annotations

import ast
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

Method = Callable[..., np.ndarray]
_REGISTRY: dict[str, Method] = {}


def register(name: str):
    def deco(fn: Method) -> Method:
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return list(_REGISTRY)


def parse_spec(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    params = {}
    for item in filter(None, rest.split(",")):
        key, _, val = item.partition("=")
        try:
            params[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            params[key] = val
    return name, params


def run(spec: str, X: np.ndarray, seed: int = 0) -> np.ndarray:
    name, params = parse_spec(spec)
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    Y = np.asarray(_REGISTRY[name](X, seed=seed, **params), dtype=np.float64)
    if Y.shape != (X.shape[0], 2):
        raise ValueError(f"{spec} returned shape {Y.shape}, expected {(X.shape[0], 2)}")
    return Y


@register("random")
def random_layout(X, seed=0):
    """Chance-level reference: every metric should be near its floor."""
    return np.random.default_rng(seed).normal(size=(X.shape[0], 2))


@register("pca")
def pca(X, seed=0):
    return PCA(n_components=2, random_state=seed).fit_transform(X)


@register("mds")
def mds(X, seed=0, max_n=3000, **kw):
    """Metric MDS (SMACOF). Skipped above max_n points because it is O(N^2) per iteration."""
    if X.shape[0] > max_n:
        raise RuntimeError(f"mds skipped: N={X.shape[0]} > max_n={max_n}")
    return MDS(n_components=2, n_init=1, init="classical_mds", random_state=seed, **kw).fit_transform(X)


@register("tsne")
def tsne(X, seed=0, perplexity=30.0, **kw):
    from openTSNE import TSNE

    perplexity = min(perplexity, (X.shape[0] - 1) / 3)
    return np.asarray(TSNE(perplexity=perplexity, random_state=seed, n_jobs=-1, **kw).fit(X))


@register("umap")
def umap_(X, seed=0, n_neighbors=15, min_dist=0.1, **kw):
    import umap

    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed, **kw).fit_transform(X)


@register("densmap")
def densmap(X, seed=0, n_neighbors=15, **kw):
    import umap

    return umap.UMAP(n_neighbors=n_neighbors, densmap=True, random_state=seed, **kw).fit_transform(X)


# NOTE: the annoy wheel installed here (1.17.3, macOS arm64) returns wrong
# neighbours (e.g. 1 instead of k), silently corrupting any method that builds
# its kNN graph with it.  PaCMAP therefore uses faiss, and TriMap gets an exact
# kNN graph from scikit-learn.  tests/test_methods.py guards both.


@register("pacmap")
def pacmap_(X, seed=0, n_neighbors=10, **kw):
    import pacmap

    model = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed, knn_backend="faiss", **kw)
    return model.fit_transform(np.ascontiguousarray(X, dtype=np.float32), init="pca")


@register("trimap")
def trimap_(X, seed=0, n_inliers=12, **kw):
    import trimap

    k = min(n_inliers + 50, X.shape[0])  # what TriMap requests from annoy itself
    dist, nbrs = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)  # self in column 0
    # TRIMAP has no seed argument; it draws from NumPy's global RNG.
    np.random.seed(seed)
    model = trimap.TRIMAP(n_dims=2, n_inliers=n_inliers, knn_tuple=(nbrs, dist), **kw)
    return model.fit_transform(X)


@register("scalerank")
def scalerank(X, seed=0, **kw):
    """Prototype of direction A (docs/scalerank.md); e.g. ``scalerank:gamma=0.5``."""
    from .proto.scalerank import ScaleRank

    return ScaleRank(random_state=seed, **kw).fit_transform(X)


@register("scalerank_fast")
def scalerank_fast(X, seed=0, **kw):
    """ScaleRank without N x N rank tables (kNN + multi-scale pool + fresh candidates)."""
    from .proto.scalerank_fast import ScaleRankFast

    return ScaleRankFast(random_state=seed, **kw).fit_transform(X)


@register("scalerank_rs")
def scalerank_rs(X, seed=0, **kw):
    """Rust port of ScaleRankFast (scalerank-web's scalerank-core, via the scalerank_rs bindings)."""
    import scalerank_rs

    return scalerank_rs.ScaleRankFast(random_state=seed, **kw).fit_transform(np.ascontiguousarray(X, dtype=np.float64))


@register("umap_rs")
def umap_rs(X, seed=0, **kw):
    """Rust UMAP from scalerank-web (umap-learn's defaults, PCA initialisation)."""
    import scalerank_rs

    return scalerank_rs.umap(np.ascontiguousarray(X, dtype=np.float64), random_state=seed, **kw)


@register("tsne_rs")
def tsne_rs(X, seed=0, **kw):
    """Rust t-SNE from scalerank-web (openTSNE's defaults, Barnes-Hut)."""
    import scalerank_rs

    return scalerank_rs.tsne(np.ascontiguousarray(X, dtype=np.float64), random_state=seed, **kw)
