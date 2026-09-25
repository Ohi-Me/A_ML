"""Blend two stage-1 runs into one out-of-fold score table (input for stage 2).

  python code/xgboost/blend_oof.py --full xgb_c2_sim19_v2 --noemb xgb_c2_sim19_noemb_v2 --w 0.75 --out xgb_c2_blend_v2

Training S1 all come from countries with labels, so one weight is used here. At test time the final predictor uses
w_seen for countries present in training and w_unseen (0.5) for countries without any training label."""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import score_split  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--full", required=True)
ap.add_argument("--noemb", required=True)
ap.add_argument("--w", type=float, default=0.75)
ap.add_argument("--w_unseen", type=float, default=0.5)
ap.add_argument("--out", required=True)
ap.add_argument("--simdrop", type=int, default=19)
args = ap.parse_args()
gpu_init()
A = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.full}.parquet"))
B = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.noemb}.parquet"), columns=["a", "b", "p"]).rename({"p": "p2"})
D = A.join(B, on=["a", "b"], how="left").with_columns(
    (args.w * pl.col("p") + (1 - args.w) * pl.col("p2").fill_null(pl.col("p"))).alias("p")).drop("p2")
D.write_parquet(os.path.join(CACHE, "feats", f"oof_{args.out}.parquet"))
rdir = os.path.join(RESULTS, "xgboost", args.out)
os.makedirs(rdir, exist_ok=True)
dm = np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None
res = score_split(D, "train", rdir, tune_folds=(1, 2), drop=dm)
rep = {"blend": {"full": args.full, "noemb": args.noemb, "w_seen": args.w, "w_unseen": args.w_unseen},
       "val": res["val"], "best": res["best"]}
print(json.dumps(rep), flush=True)
json.dump(rep, open(os.path.join(rdir, "report.json"), "w"), indent=1)
