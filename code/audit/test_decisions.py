"""Per-country decisions of one test score table (isotonic from an OOF table, expected-F0.5 decoding).

  python code/audit/test_decisions.py --scores gnn/gnn_cfA_test_scores.parquet --col gnn --logit --oof oof_gnn_mlx_v2.parquet --name cfA
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from audit.gap_audit import OUT, decision_stats, decode, iso_from  # noqa: E402
from common.decide import ids  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--col", default="p")
ap.add_argument("--logit", action="store_true")
ap.add_argument("--oof", required=True)
ap.add_argument("--name", required=True)
args = ap.parse_args()
gpu_init()
os.makedirs(OUT, exist_ok=True)
T = pl.read_parquet(os.path.join(CACHE, args.scores)).select(["a", "b", args.col]).rename({args.col: "p"})
if args.logit:
    T = T.with_columns((1 / (1 + (-pl.col("p")).exp())).cast(pl.Float32).alias("p"))
keep, A = decode(T, iso_from(args.oof))
c1 = ids("test")[0]["country"].to_numpy()
st = decision_stats(keep, A, c1, np.ones(len(c1), bool))
st["_total_matches"] = keep.height
print(json.dumps(st, indent=1), flush=True)
json.dump(st, open(os.path.join(OUT, f"test_decisions_{args.name}.json"), "w"), indent=1)
