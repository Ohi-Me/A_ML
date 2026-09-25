"""Label-free transfer check for the unseen country: score the test candidates with two stage-1 models (with and
without embedding features) and compare their decisions per country. If the embedding misbehaves on a new
language, the two models should disagree much more there than on the countries seen in training."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import best_per_record, decide_records, ids  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--test_feats", default="c2_test")
ap.add_argument("--tags", default="xgb_c2_sim19,xgb_c2_sim19_noemb")
ap.add_argument("--rename", default="r_e2f_emb:r_e3_emb")
args = ap.parse_args()
gpu_init()
ren = dict(x.split(":") for x in args.rename.split(",") if x)
files = sorted(glob.glob(os.path.join(CACHE, "feats", args.test_feats, "part_*.parquet")))
scores = {}
for tag in args.tags.split(","):
    rep = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
    m = xgb.Booster(model_file=os.path.join(RESULTS, "xgboost", tag, "model_A.json"))
    outs = []
    for f in files:
        d = pl.read_parquet(f)
        d = d.rename({k: v for k, v in ren.items() if k in d.columns})
        X = d.select([pl.col(c).cast(pl.Float32) for c in rep["features"]]).to_numpy()
        outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", m.predict(xgb.DMatrix(X)))))
    D = pl.concat(outs)
    dec = json.load(open(os.path.join(RESULTS, "xgboost", tag, "decision.json")))["best"]
    scores[tag] = decide_records(best_per_record(D), dec["thr"], dec["margin"]).with_columns(pl.lit(1).alias(tag))
t1, t2 = args.tags.split(",")
J = scores[t1].join(scores[t2], on=["a", "b"], how="full", coalesce=True).with_columns(
    pl.col(t1).fill_null(0), pl.col(t2).fill_null(0))
cty = ids("test")[0]["country"].to_numpy()
J = J.with_columns(pl.Series("country", cty[J["a"].to_numpy()]))
out = {}
for c in sorted(set(cty.tolist())):
    x = J.filter(pl.col("country") == c)
    both = int(((x[t1] == 1) & (x[t2] == 1)).sum())
    only1 = int(((x[t1] == 1) & (x[t2] == 0)).sum())
    only2 = int(((x[t1] == 0) & (x[t2] == 1)).sum())
    out[c] = {"both": both, f"only_{t1}": only1, f"only_{t2}": only2,
              "disagree_rate": (only1 + only2) / max(both + only1 + only2, 1)}
    print(c, out[c], flush=True)
json.dump(out, open(os.path.join(RESULTS, "xgboost", t2, "test_agreement.json"), "w"), indent=1)
