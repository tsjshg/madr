"""Run methods x datasets x seeds, cache results, and aggregate them.

Layout under ``results/`` (git-ignored)::

    embeddings/<dataset>/<method>__s<seed>.npy   2-D embedding
    metrics/<dataset>/<method>__s<seed>.json     scalar metrics + runtime
    curves/<dataset>/<method>__s<seed>.npz       R_NX(K) and per-point overlap
    datasets/<dataset>.json                      reference values (k-NN acc in X)

Embeddings are reused when present, so adding a method or a dataset only
computes what is missing.  ``--force`` recomputes embeddings,
``--rescore`` recomputes metrics from cached embeddings (after a metric change).
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import datasets, methods
from .metrics import evaluate, knn_accuracy

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", spec)


def _paths(ds: str, spec: str, seed: int) -> dict[str, Path]:
    stem = f"{slug(spec)}__s{seed}"
    return {
        "emb": RESULTS_DIR / "embeddings" / ds / f"{stem}.npy",
        "metrics": RESULTS_DIR / "metrics" / ds / f"{stem}.json",
        "curves": RESULTS_DIR / "curves" / ds / f"{stem}.npz",
    }


def _write_dataset_reference(ds: datasets.Dataset) -> None:
    path = RESULTS_DIR / "datasets" / f"{ds.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    ref = {"name": ds.name, "n": ds.n, "dim": ds.X.shape[1], "description": ds.description}
    if ds.labels is not None:
        ref["knn_acc_in_X"] = knn_accuracy(ds.X, ds.labels)
    if ds.labels_coarse is not None:
        ref["knn_acc_coarse_in_X"] = knn_accuracy(ds.X, ds.labels_coarse)
    path.write_text(json.dumps(ref, indent=2))


def run_one(ds: datasets.Dataset, spec: str, seed: int, force=False, rescore=False, log=print) -> dict:
    p = _paths(ds.name, spec, seed)
    for f in p.values():
        f.parent.mkdir(parents=True, exist_ok=True)
    if p["metrics"].exists() and p["emb"].exists() and not (force or rescore):
        return json.loads(p["metrics"].read_text())

    record = {"dataset": ds.name, "method": spec, "seed": seed, "n": ds.n}
    if p["emb"].exists() and not force:
        Y = np.load(p["emb"])
        prev = json.loads(p["metrics"].read_text()) if p["metrics"].exists() else {}
        record["time_s"] = prev.get("time_s", float("nan"))
    else:
        t0 = time.perf_counter()
        try:
            Y = methods.run(spec, ds.X, seed=seed)
        except Exception as e:  # keep going; record why this cell is empty
            record.update(status="error", error=f"{type(e).__name__}: {e}")
            log(f"  ! {ds.name} {spec} s{seed}: {record['error']}")
            if not isinstance(e, RuntimeError):
                traceback.print_exc()
            p["metrics"].write_text(json.dumps(record, indent=2))
            return record
        record["time_s"] = time.perf_counter() - t0
        np.save(p["emb"], Y)

    t0 = time.perf_counter()
    scalars, arrays = evaluate(ds.X, Y, ds.labels)
    if ds.labels_coarse is not None:
        scalars["knn_acc_coarse"] = knn_accuracy(Y, ds.labels_coarse)
    record.update(scalars)
    record["status"] = "ok"
    record["eval_time_s"] = time.perf_counter() - t0
    np.savez_compressed(p["curves"], **arrays)
    p["metrics"].write_text(json.dumps(record, indent=2))
    log(f"  {ds.name:14s} {spec:24s} s{seed}  {record['time_s']:7.1f}s  "
        f"R_NX@10={record.get('rnx@10', float('nan')):.3f}  AUC={record['auc_rnx']:.3f}")
    return record


def run_grid(ds_names, specs, seeds, max_n=None, force=False, rescore=False, log=print) -> pd.DataFrame:
    rows = []
    for name in ds_names:
        ds = datasets.load(name, max_n=max_n)
        log(f"[{name}] N={ds.n} dim={ds.X.shape[1]}")
        _write_dataset_reference(ds)
        for spec in specs:
            for seed in seeds:
                rows.append(run_one(ds, spec, seed, force=force, rescore=rescore, log=log))
    return pd.DataFrame(rows)


def load_results() -> pd.DataFrame:
    rows = [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / "metrics").glob("*/*.json"))]
    return pd.DataFrame(rows)


SUMMARY_COLS = [
    "rnx@10", "trust@10", "cont@10", "auc_rnx", "dist_spearman",
    "density_corr", "knn_acc", "knn_acc_coarse", "footrule", "time_s",
]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds for each (dataset, method)."""
    ok = df[df.get("status", "ok") == "ok"]
    cols = [c for c in SUMMARY_COLS if c in ok.columns]
    return ok.groupby(["dataset", "method"], sort=False)[cols].mean()
