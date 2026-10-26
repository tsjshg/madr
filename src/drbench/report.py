"""Figures and tables from cached results.

- ``rnx_<dataset>.png``: R_NX(K) against log K, one line per method.  Reading
  left to right goes from local to global structure; the legend gives AUC.
- ``scatter_<dataset>.png``: the embeddings side by side (seed 0).
- ``summary.md``: mean over seeds of the headline metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import datasets
from .runner import RESULTS_DIR, _paths, load_results, summarize

# Categorical slots in fixed order (validated default palette, light mode).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "^", "s", "D"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Colour follows the method, never its rank: known methods get fixed slots.
METHOD_SLOTS = {"pca": 0, "tsne": 1, "umap": 2, "densmap": 3, "pacmap": 4, "mds": 5, "trimap": 6}
REFERENCE_STYLE = {"color": "#9a9893", "linestyle": "--"}  # the random layout
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def _method_style(spec: str, extra: dict[str, int]) -> dict:
    base = spec.split(":")[0]
    if base == "random":
        return dict(REFERENCE_STYLE)
    if spec in extra:
        slot = extra[spec]
    elif base in METHOD_SLOTS and spec == base:
        slot = METHOD_SLOTS[base]
    else:  # variants and new methods take the remaining slots in order
        slot = 7 + len(extra)
        extra[spec] = slot
    return {"color": SLOTS[slot % len(SLOTS)], "linestyle": "-" if slot < len(SLOTS) else ":"}


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_rnx(dataset: str, specs: list[str], out: Path, seed: int = 0) -> None:
    df = load_results()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    extra: dict[str, int] = {}
    for spec in specs:
        f = _paths(dataset, spec, seed)["curves"]
        if not f.exists():
            continue
        r = np.load(f)["r_nx"][1:]
        k = np.arange(1, len(r) + 1)
        auc = df[(df.dataset == dataset) & (df.method == spec)]["auc_rnx"].mean()
        ax.plot(k, r, linewidth=2, label=f"{spec}  (AUC {auc:.3f})", **_method_style(spec, extra))
    ax.set_xscale("log")
    ax.set_xlabel("K (neighbourhood size)  →  more global", color=TEXT, fontsize=9)
    ax.set_ylabel("R_NX(K)", color=TEXT, fontsize=9)
    ax.set_title(f"{dataset}: neighbourhood preservation by scale", color=TEXT, fontsize=10, loc="left")
    _style_axes(ax)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_scatter(dataset: str, specs: list[str], out: Path, seed: int = 0, max_n: int | None = None) -> None:
    ds = datasets.load(dataset, max_n=max_n)
    shown = [s for s in specs if _paths(dataset, s, seed)["emb"].exists()]
    if not shown:
        return
    ncol = min(4, len(shown))
    nrow = int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow), dpi=150, squeeze=False)
    categorical = ds.labels is not None and np.array_equal(ds.color, ds.labels)
    for ax, spec in zip(axes.flat, shown):
        Y = np.load(_paths(dataset, spec, seed)["emb"])
        size = max(2, min(12, 4000 / ds.n))
        if categorical:
            # >8 classes: hue x marker composite encoding instead of new hues
            for c in np.unique(ds.labels):
                m = ds.labels == c
                ax.scatter(Y[m, 0], Y[m, 1], s=size, color=SLOTS[c % 8],
                           marker=MARKERS[(c // 8) % len(MARKERS)], linewidths=0, alpha=0.85)
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=size, c=ds.color, cmap=SEQUENTIAL, linewidths=0)
        ax.set_title(spec, color=TEXT, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ax.spines.values():
            side.set_color(GRID)
    for ax in list(axes.flat)[len(shown):]:
        ax.axis("off")
    if categorical:
        names = ds.label_names or [str(c) for c in np.unique(ds.labels)]
        handles = [plt.Line2D([], [], linestyle="", marker=MARKERS[(i // 8) % len(MARKERS)],
                              color=SLOTS[i % 8], label=n) for i, n in enumerate(names)]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(names), 8), fontsize=7,
                   frameon=False, labelcolor=TEXT)
    fig.suptitle(f"{dataset} (seed {seed})", color=TEXT, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06 if categorical else 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def write_summary(out: Path) -> None:
    df = load_results()
    table = summarize(df)
    lines = [
        "# Benchmark summary",
        "",
        "Mean over seeds. Higher is better except `footrule` and `time_s`.",
        "`rnx@10`/`trust@10`/`cont@10`: local; `auc_rnx`: all scales (log-K weighted);",
        "`dist_spearman`: global distances; `density_corr`: local density;",
        "`knn_acc(_coarse)`: label (super-cluster) separability in the embedding.",
        "",
    ]
    for name, sub in table.groupby(level=0, sort=False):
        ref_file = RESULTS_DIR / "datasets" / f"{name}.json"
        lines.append(f"## {name}")
        if ref_file.exists():
            import json

            ref = json.loads(ref_file.read_text())
            extra = f", k-NN acc in X = {ref['knn_acc_in_X']:.3f}" if "knn_acc_in_X" in ref else ""
            lines.append(f"{ref['description']} (N={ref['n']}, dim={ref['dim']}{extra})")
        lines.append("")
        lines.append(sub.droplevel(0).round(3).to_markdown())
        lines.append("")
    errors = df[df.get("status") == "error"] if "status" in df else df.iloc[0:0]
    if len(errors):
        lines.append("## Skipped / failed")
        for _, r in errors.iterrows():
            lines.append(f"- {r.dataset} / {r.method} / s{r.seed}: {r.error}")
    out.write_text("\n".join(lines) + "\n")


def build(specs: list[str] | None = None, max_n: int | None = None) -> Path:
    df = load_results()
    out = RESULTS_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    for dataset in df.dataset.unique():
        ds_specs = specs or list(dict.fromkeys(df[df.dataset == dataset].method))
        plot_rnx(dataset, ds_specs, out / f"rnx_{dataset}.png")
        plot_scatter(dataset, ds_specs, out / f"scatter_{dataset}.png", max_n=max_n)
    write_summary(out / "summary.md")
    return out
