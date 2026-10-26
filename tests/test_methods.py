import numpy as np
import pytest

from drbench import datasets, methods


def test_parse_spec():
    assert methods.parse_spec("tsne") == ("tsne", {})
    assert methods.parse_spec("umap:n_neighbors=50,min_dist=0.5,metric=cosine") == (
        "umap",
        {"n_neighbors": 50, "min_dist": 0.5, "metric": "cosine"},
    )


@pytest.mark.parametrize("spec", ["random", "pca", "tsne", "umap", "densmap", "pacmap", "trimap"])
def test_methods_return_2d(spec):
    X = datasets.load("hier_blobs", max_n=200).X
    Y = methods.run(spec, X, seed=0)
    assert Y.shape == (200, 2)
    assert np.isfinite(Y).all()


def test_synthetic_datasets_are_deterministic():
    a, b = datasets.load("hier_blobs"), datasets.load("hier_blobs")
    assert np.array_equal(a.X, b.X)


@pytest.mark.parametrize("spec", ["tsne", "umap", "pacmap", "trimap"])
def test_neighbour_methods_beat_chance_clearly(spec):
    """Catches silently broken kNN backends (see the annoy note in methods.py)."""
    from drbench.metrics import knn_accuracy

    ds = datasets.load("digits")
    Y = methods.run(spec, ds.X, seed=0)
    assert knn_accuracy(Y, ds.labels) > 0.8
