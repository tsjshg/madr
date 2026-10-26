"""Command line entry point: ``uv run drbench {list,run,report}``."""

from __future__ import annotations

import argparse

from . import datasets, methods, report, runner

DEFAULT_DATASETS = ["s_curve", "hier_blobs", "density_blobs", "digits", "pollen", "usoskin", "kolodziejczyk", "buettner"]
DEFAULT_METHODS = ["random", "pca", "mds", "tsne", "umap", "densmap", "pacmap", "trimap"]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="drbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list datasets and methods")

    r = sub.add_parser("run", help="compute embeddings and metrics (cached)")
    r.add_argument("-d", "--datasets", nargs="+", default=DEFAULT_DATASETS)
    r.add_argument("-m", "--methods", nargs="+", default=DEFAULT_METHODS,
                   help="method specs, e.g. tsne or tsne:perplexity=5")
    r.add_argument("-s", "--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--max-n", type=int, default=None, help="subsample datasets larger than this")
    r.add_argument("--force", action="store_true", help="recompute embeddings")
    r.add_argument("--rescore", action="store_true", help="recompute metrics from cached embeddings")

    rep = sub.add_parser("report", help="write figures and summary.md to results/report/")
    rep.add_argument("-m", "--methods", nargs="+", default=None)

    a = p.parse_args(argv)
    if a.cmd == "list":
        print("datasets:", ", ".join(datasets.available()))
        print("methods: ", ", ".join(methods.available()))
    elif a.cmd == "run":
        runner.run_grid(a.datasets, a.methods, a.seeds, max_n=a.max_n, force=a.force, rescore=a.rescore)
        print(runner.summarize(runner.load_results()).round(3).to_string())
    elif a.cmd == "report":
        print(f"written to {report.build(a.methods)}")


if __name__ == "__main__":
    main()
