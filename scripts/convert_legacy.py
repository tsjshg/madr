"""Convert the 2020 scRNA-seq pickles (../*.pkl) into data/raw/<name>.npz.

The pickles hold a ``scRNAData`` namedtuple defined in a notebook's
``__main__``; we recreate that type so they unpickle, then store sparse counts
and integer labels in a pickle-free format.

    uv run python scripts/convert_legacy.py [legacy_dir]
"""

import collections
import pickle
import sys
from pathlib import Path

import __main__
import numpy as np
from scipy import sparse

scRNAData = collections.namedtuple("scRNAData", ["name", "data", "target", "target_name", "meta_data"])
__main__.scRNAData = scRNAData

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["pollen", "usoskin", "kolodziejczyk", "buettner"]


def main(legacy_dir: Path) -> None:
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        with open(legacy_dir / f"{name}.pkl", "rb") as f:
            d = pickle.load(f)
        counts = sparse.csr_matrix(d.data.values.astype(np.float32))
        labels = np.asarray(d.target, dtype=np.int64)
        np.savez_compressed(
            out / f"{name}.npz",
            data=counts.data,
            indices=counts.indices,
            indptr=counts.indptr,
            shape=np.array(counts.shape),
            labels=labels,
            label_names=np.array([str(s) for s in d.target_name]),
        )
        print(f"{name}: {counts.shape}, {len(set(labels))} labels -> {out / name}.npz")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent)
