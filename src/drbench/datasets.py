"""Benchmark datasets.

Every loader returns a :class:`Dataset` whose ``X`` is the reference
high-dimensional space: it is both the input to every method and the space
the metrics compare against.  For scRNA-seq data that space is the usual
log-normalised, highly-variable-gene, 50-PC representation, so methods are
judged against what practitioners actually feed into t-SNE / UMAP.

Real datasets are cached under ``data/`` (git-ignored):
- ``data/raw/<name>.npz``: sparse counts + labels, made by
  ``scripts/convert_legacy.py`` from the 2020 pickles.
- ``data/openml/``: MNIST-style data fetched from OpenML on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import sparse
from sklearn import datasets as skd
from sklearn.decomposition import PCA

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    labels: np.ndarray | None = None  # categorical, used for k-NN accuracy
    color: np.ndarray | None = None  # what to colour scatter plots by
    labels_coarse: np.ndarray | None = None  # higher-level groups, if any
    description: str = ""
    label_names: list[str] | None = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return self.X.shape[0]

    def __post_init__(self):
        if self.color is None and self.labels is not None:
            self.color = self.labels


_REGISTRY: dict[str, Callable[..., Dataset]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    return list(_REGISTRY)


def load(name: str, max_n: int | None = None, seed: int = 0) -> Dataset:
    """Load a dataset, optionally taking a uniform random subsample of max_n points."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    ds = _REGISTRY[name]()
    if max_n is not None and ds.n > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(ds.n, max_n, replace=False))
        ds = Dataset(
            name=ds.name,
            X=ds.X[idx],
            labels=None if ds.labels is None else ds.labels[idx],
            color=None if ds.color is None else ds.color[idx],
            labels_coarse=None if ds.labels_coarse is None else ds.labels_coarse[idx],
            description=f"{ds.description} (subsampled to {max_n})",
            label_names=ds.label_names,
        )
    return ds


# ---------------------------------------------------------------- synthetic


@register("s_curve")
def s_curve() -> Dataset:
    X, t = skd.make_s_curve(2000, noise=0.05, random_state=0)
    return Dataset("s_curve", X, color=t, description="3-D S-curve, 2000 pts (manifold unrolling)")


@register("swiss_roll")
def swiss_roll() -> Dataset:
    X, t = skd.make_swiss_roll(2000, noise=0.05, random_state=0)
    return Dataset("swiss_roll", X, color=t, description="3-D Swiss roll, 2000 pts")


@register("hier_blobs")
def hier_blobs() -> Dataset:
    """3 super-clusters x 4 sub-clusters x 100 pts in 50-D.

    Tests whether the arrangement *between* clusters (global structure)
    survives: sub-clusters of the same super-cluster should stay together.
    """
    rng = np.random.default_rng(0)
    dim, n_super, n_sub, n_pts = 50, 3, 4, 100
    X, sub, sup = [], [], []
    for s in range(n_super):
        c_super = rng.normal(scale=10.0, size=dim)
        for b in range(n_sub):
            c_sub = c_super + rng.normal(scale=3.0, size=dim)
            X.append(c_sub + rng.normal(size=(n_pts, dim)))
            sub += [s * n_sub + b] * n_pts
            sup += [s] * n_pts
    return Dataset(
        "hier_blobs",
        np.vstack(X),
        labels=np.array(sub),
        labels_coarse=np.array(sup),
        description="3 super x 4 sub Gaussian clusters, 1200 pts, 50-D (global structure)",
    )


@register("density_blobs")
def density_blobs() -> Dataset:
    """4 well-separated clusters with spreads 0.5, 1, 2, 4 in 10-D.

    Tests whether relative cluster density is preserved (t-SNE / UMAP
    equalise it; densMAP was designed not to).
    """
    rng = np.random.default_rng(0)
    dim, n_pts = 10, 250
    X, lab = [], []
    for i, s in enumerate([0.5, 1.0, 2.0, 4.0]):
        center = np.zeros(dim)
        center[i] = 40.0
        X.append(center + rng.normal(scale=s, size=(n_pts, dim)))
        lab += [i] * n_pts
    return Dataset(
        "density_blobs",
        np.vstack(X),
        labels=np.array(lab),
        description="4 clusters with spreads 0.5/1/2/4, 1000 pts, 10-D (density)",
    )


@register("digits")
def digits() -> Dataset:
    d = skd.load_digits()
    return Dataset("digits", d.data.astype(float), labels=d.target, description="sklearn 8x8 digits, 1797 x 64")


# ---------------------------------------------------------------- scRNA-seq


def preprocess_counts(
    counts: sparse.spmatrix | np.ndarray, n_hvg: int = 2000, n_pcs: int = 50, target_sum: float = 1e4
) -> np.ndarray:
    """Library-size normalise, log1p, keep the most variable genes, PCA.

    A plain-NumPy version of the standard Scanpy recipe (without scaling).
    """
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    norm = sparse.diags(target_sum / lib) @ counts
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(axis=0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(axis=0)).ravel() - mean**2
    hvg = np.argsort(var)[::-1][: min(n_hvg, counts.shape[1])]
    dense = norm[:, hvg].toarray()
    n_pcs = min(n_pcs, *dense.shape)
    return PCA(n_components=n_pcs, random_state=0).fit_transform(dense)


def _load_raw(name: str, description: str) -> Dataset:
    path = DATA_DIR / "raw" / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/convert_legacy.py first")
    z = np.load(path, allow_pickle=False)
    counts = sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    label_names = [str(s) for s in z["label_names"]]
    return Dataset(
        name,
        preprocess_counts(counts),
        labels=z["labels"],
        description=description,
        label_names=label_names,
    )


@register("pollen")
def pollen() -> Dataset:
    return _load_raw("pollen", "Pollen et al. 2014, 301 cells, 11 cell types")


@register("usoskin")
def usoskin() -> Dataset:
    return _load_raw("usoskin", "Usoskin et al. 2015, 622 neurons, 4 types")


@register("kolodziejczyk")
def kolodziejczyk() -> Dataset:
    return _load_raw("kolodziejczyk", "Kolodziejczyk et al. 2015, 704 mESCs, 3 culture conditions")


@register("buettner")
def buettner() -> Dataset:
    return _load_raw("buettner", "Buettner et al. 2015, 288 mESCs, 3 cell-cycle phases")


# ---------------------------------------------------------------- OpenML


def _openml(name: str, openml_name: str, description: str, n: int = 10_000) -> Dataset:
    """Fetch from OpenML (network on first use) and keep a fixed 10k subset."""
    cache = DATA_DIR / "openml" / f"{name}_{n}.npz"
    if not cache.exists():
        d = skd.fetch_openml(openml_name, version=1, as_frame=False, data_home=DATA_DIR / "openml")
        idx = np.sort(np.random.default_rng(0).choice(len(d.target), n, replace=False))
        X = d.data[idx].astype(np.float32) / 255.0
        y = d.target[idx].astype(int)
        X50 = PCA(n_components=50, random_state=0).fit_transform(X)
        np.savez_compressed(cache, X=X50, y=y)
    z = np.load(cache)
    return Dataset(name, z["X"].astype(np.float64), labels=z["y"], description=description)


@register("mnist10k")
def mnist10k() -> Dataset:
    return _openml("mnist10k", "mnist_784", "MNIST, 10k random subset, PCA-50")


@register("fmnist10k")
def fmnist10k() -> Dataset:
    return _openml("fmnist10k", "Fashion-MNIST", "Fashion-MNIST, 10k random subset, PCA-50")
