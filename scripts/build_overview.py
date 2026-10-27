"""Build docs/overview.html, the illustrated Japanese overview of the project.

Pulls the numbers the charts need from ``results/`` and injects them into
``docs/overview_template.html``.  Needed runs (``drbench run -m ...``): the
baselines, the current ScaleRank default at gamma 1/0.5/0, and the first
prototype's runs under their explicit names (see ALIASES).  Figures quoted
from one-off investigations (2022 notebook scores, force decomposition,
hubness, the Q sweep) are written in the template itself; see
docs/scalerank.md for where each comes from.

    uv run python scripts/build_overview.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from drbench import datasets
from drbench.metrics import local_radius
from drbench.runner import _paths, load_results, summarize

ROOT = Path(__file__).resolve().parents[1]
BASE = ["pca", "mds", "tsne", "umap", "densmap", "pacmap", "trimap"]
FIRST = "hard=0.3,n_neg=1,per_point=32"  # the first prototype: triplets + hard negatives
# Short keys used by the page -> drbench method specs.
ALIASES = {
    "sr:1.0": "scalerank:gamma=1.0",  # current default (softmax, n_neg=20, per_point=8, hard=0)
    "sr:0.5": "scalerank:gamma=0.5",
    "sr:0.0": "scalerank:gamma=0.0",
    "v1:1.0": f"scalerank:gamma=1.0,{FIRST}",
    "v1:0.5": f"scalerank:gamma=0.5,{FIRST}",
    "v1:0.0": f"scalerank:gamma=0.0,{FIRST}",
    "v1:nohard": "scalerank:gamma=1.0,hard=0.0,n_neg=1,per_point=32",
    "v1:R": "scalerank:gamma=1.0,local_scaling=10,local_scaling_kmax=30,hard=0.15,n_neg=1,per_point=32",
    "q100": "scalerank:gamma=1.0,hard=0.0,n_neg=100,per_point=4",
}
ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
METRICS = ["rnx@10", "auc_rnx", "trust@10", "dist_spearman", "density_corr", "knn_acc", "time_s"]


def spec(key):
    return ALIASES.get(key, key)


def unit_square(Y):
    Y = Y - Y.min(0)
    return np.round(Y / Y.max(), 3).tolist()


def outlier_mask(Y):
    r = local_radius(Y, 10)
    return r / np.median(r) > 3


def k_occurrence(X, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return np.bincount(idx.ravel(), minlength=len(X))


def emb(ds, key, seed=0):
    return np.load(_paths(ds, spec(key), seed)["emb"])


def main():
    table = summarize(load_results()).reset_index()
    summary = {}
    for ds in ORDER:
        summary[ds] = {}
        for key in BASE + list(ALIASES) + ["random"]:
            r = table[(table.dataset == ds) & (table.method == spec(key))]
            if r.empty:
                continue
            r = r.iloc[0]
            summary[ds][key] = {k: (None if k not in r or r[k] != r[k] else round(float(r[k]), 3)) for k in METRICS}

    curves = {}
    for ds in ["pollen", "digits"]:
        n = datasets.load(ds).n
        grid = np.unique(np.round(np.geomspace(1, n - 2, 70)).astype(int))
        curves[ds] = {}
        for key in BASE + ["sr:1.0", "sr:0.0", "v1:1.0"]:
            r = np.load(_paths(ds, spec(key), 0)["curves"])["r_nx"]
            curves[ds][key] = {"k": grid.tolist(), "r": [round(float(r[k]), 4) for k in grid]}

    digits = datasets.load("digits")
    hier = datasets.load("hier_blobs")
    hier_keys = ["v1:1.0", "v1:nohard", "sr:1.0", "tsne"]
    embeddings = {
        "digits": {
            "labels": digits.labels.tolist(),
            "emb": {k: unit_square(emb("digits", k)) for k in ["mds", "tsne", "v1:1.0", "sr:1.0"]},
        },
        "hier_blobs": {
            "labels": hier.labels_coarse.tolist(),
            "antihub": (k_occurrence(hier.X) <= 2).astype(int).tolist(),
            "emb": {k: unit_square(emb("hier_blobs", k)) for k in hier_keys},
            "outlier": {k: outlier_mask(emb("hier_blobs", k)).astype(int).tolist() for k in hier_keys},
        },
    }

    outfrac = {}
    for ds in ORDER:
        outfrac[ds] = {}
        for key in ["v1:1.0", "v1:0.5", "v1:0.0", "v1:R", "sr:1.0", "sr:0.5", "sr:0.0", "q100", "tsne", "mds"]:
            fr = [outlier_mask(np.load(p)).mean() for s in (0, 1, 2) if (p := _paths(ds, spec(key), s)["emb"]).exists()]
            outfrac[ds][key] = round(float(np.mean(fr)), 4) if fr else None

    data = {"summary": summary, "curves": curves, "emb": embeddings, "outfrac": outfrac}
    template = (ROOT / "docs" / "overview_template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "docs" / "overview.html"
    out.write_text(html)
    print(f"written {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
