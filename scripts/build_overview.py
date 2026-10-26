"""Build docs/overview.html, the illustrated Japanese overview of the project.

Pulls the numbers the charts need from ``results/`` (run ``drbench run`` for
the baselines and the ScaleRank variants first) and injects them into
``docs/overview_template.html``.  Figures quoted from one-off investigations
(2022 notebook scores, force decomposition, hubness) are written in the
template itself; see docs/scalerank.md for where each comes from.

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
SR = ["scalerank:gamma=1.0", "scalerank:gamma=0.5", "scalerank:gamma=0.0"]
VARIANT_R = "scalerank:gamma=1.0,local_scaling=10,local_scaling_kmax=30,hard=0.15"
ORDER = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
METRICS = ["rnx@10", "auc_rnx", "trust@10", "dist_spearman", "density_corr", "knn_acc", "time_s"]


def unit_square(Y):
    Y = Y - Y.min(0)
    return np.round(Y / Y.max(), 3).tolist()


def outlier_mask(Y):
    r = local_radius(Y, 10)
    return r / np.median(r) > 3


def k_occurrence(X, k=10):
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    return np.bincount(idx.ravel(), minlength=len(X))


def main():
    table = summarize(load_results()).reset_index()
    summary = {}
    for ds in ORDER:
        summary[ds] = {}
        for m in BASE + SR + ["random"]:
            r = table[(table.dataset == ds) & (table.method == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            summary[ds][m] = {k: (None if k not in r or r[k] != r[k] else round(float(r[k]), 3)) for k in METRICS}

    curves = {}
    for ds in ["pollen", "digits"]:
        n = datasets.load(ds).n
        grid = np.unique(np.round(np.geomspace(1, n - 2, 70)).astype(int))
        curves[ds] = {}
        for m in BASE + ["scalerank:gamma=1.0", "scalerank:gamma=0.0"]:
            r = np.load(_paths(ds, m, 0)["curves"])["r_nx"]
            curves[ds][m] = {"k": grid.tolist(), "r": [round(float(r[k]), 4) for k in grid]}

    digits = datasets.load("digits")
    hier = datasets.load("hier_blobs")
    hier_methods = ["scalerank:gamma=1.0", "scalerank:gamma=1.0,hard=0.0", "tsne"]
    emb = {
        "digits": {
            "labels": digits.labels.tolist(),
            "emb": {m: unit_square(np.load(_paths("digits", m, 0)["emb"])) for m in ["mds", "tsne", "scalerank:gamma=1.0"]},
        },
        "hier_blobs": {
            "labels": hier.labels_coarse.tolist(),
            "antihub": (k_occurrence(hier.X) <= 2).astype(int).tolist(),
            "emb": {m: unit_square(np.load(_paths("hier_blobs", m, 0)["emb"])) for m in hier_methods},
            "outlier": {m: outlier_mask(np.load(_paths("hier_blobs", m, 0)["emb"])).astype(int).tolist() for m in hier_methods},
        },
    }

    outfrac = {}
    for ds in ORDER:
        outfrac[ds] = {}
        for m in ["scalerank:gamma=1.0", "tsne", VARIANT_R]:
            fr = [outlier_mask(np.load(p)).mean() for s in (0, 1, 2) if (p := _paths(ds, m, s)["emb"]).exists()]
            outfrac[ds][m] = round(float(np.mean(fr)), 4) if fr else None

    data = {"summary": summary, "curves": curves, "emb": emb, "outfrac": outfrac}
    template = (ROOT / "docs" / "overview_template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "docs" / "overview.html"
    out.write_text(html)
    print(f"written {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
