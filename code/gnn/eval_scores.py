"""Validate any re-scoring of the candidate pairs (GNN, cross-encoder fusion ...) with the standard decision rule:
threshold + runner-up margin tuned on folds 1-2, reported on fold 0 (test-prior simulation).

  python code/gnn/eval_scores.py --oof oof_xgb_c2_blend_v2.parquet --scores gnn/gnn_v2_train_scores.parquet --col gnn
"""
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
ap.add_argument("--oof", required=True)
ap.add_argument("--scores", required=True)
ap.add_argument("--col", required=True)
ap.add_argument("--logit", action="store_true", help="score column is a logit")
ap.add_argument("--out", default="gnn")
ap.add_argument("--simdrop", type=int, default=19)
args = ap.parse_args()
gpu_init()
D = pl.read_parquet(os.path.join(CACHE, "feats", args.oof))
S = pl.read_parquet(os.path.join(CACHE, args.scores))
D = D.join(S, on=["a", "b"], how="left")
sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None)
tune, val = np.isin(sc.fold, [1, 2]), sc.fold == VAL_FOLD
out = {}
for nm, col in (("stage1_input", "p"), (args.col, args.col)):
    x = D[col].to_numpy().astype(np.float64)
    if col != "p" and args.logit:
        x = 1 / (1 + np.exp(-x))
    best = best_per_record(D.select(["a", "b"]).with_columns(pl.Series("p", x)))
    grid = []
    for thr in (0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        for mg in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            m, _ = sc.metrics(decide_records(best, thr, mg), tune)
            grid.append((m["macro_f05"], thr, mg))
    f, thr, mg = max(grid)
    mv, _ = sc.metrics(decide_records(best, thr, mg), val)
    out[nm] = {"thr": thr, "margin": mg, "f05_tune": f, "f05_val": mv["macro_f05"], "fp": mv["fp"], "fn": mv["fn"]}
    print(nm, out[nm], flush=True)
os.makedirs(os.path.join(RESULTS, "gnn", args.out), exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "gnn", args.out, f"eval_{args.col}.json"), "w"), indent=1)
