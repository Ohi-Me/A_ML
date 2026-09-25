"""Does blending two stage-1 models cost anything on validation? Averages the out-of-fold probabilities of the given
runs with weight w on the first, tunes threshold + margin on folds 1-2 and reports fold 0 (test-prior simulation)."""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record, decide_records  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.split import VAL_FOLD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="xgb_c2_sim19,xgb_c2_sim19_noemb")
ap.add_argument("--simdrop", type=int, default=19)
args = ap.parse_args()
gpu_init()
t1, t2 = args.tags.split(",")
A = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{t1}.parquet"))
B = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{t2}.parquet"), columns=["a", "b", "p"]).rename({"p": "p2"})
D = A.join(B, on=["a", "b"], how="left")
sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None)
tune, val = np.isin(sc.fold, [1, 2]), sc.fold == VAL_FOLD
out = []
for w in (1.0, 0.75, 0.5, 0.25, 0.0):
    Dw = D.with_columns((w * pl.col("p") + (1 - w) * pl.col("p2")).alias("p")).select(["a", "b", "p"])
    best = best_per_record(Dw)
    grid = []
    for thr in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        for mg in (0.2, 0.3, 0.4, 0.5, 0.6):
            m, _ = sc.metrics(decide_records(best, thr, mg), tune)
            grid.append((m["macro_f05"], thr, mg))
    f, thr, mg = max(grid)
    mv, _ = sc.metrics(decide_records(best, thr, mg), val)
    out.append({"w_first": w, "thr": thr, "margin": mg, "f05_tune": f, "f05_val": mv["macro_f05"], "fp": mv["fp"], "fn": mv["fn"]})
    print(out[-1], flush=True)
json.dump(out, open(os.path.join(RESULTS, "xgboost", t2, "blend_eval.json"), "w"), indent=1)
