"""Covariate shift check for the unseen country: distributions of key features of every record's top candidate,
by country, in train and test feature tables (and the test model scores when given)."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, NORM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train_feats", default="c1_train")
ap.add_argument("--test_feats", default="c1_test")
ap.add_argument("--test_scores", default="test_scores_final_c1.parquet")
args = ap.parse_args()
gpu_init()
KEY = ["cos_e2", "cos_e3", "cos_name", "cos_addr", "b_gap2_cos_e2", "b_gap2_cos_e3", "b_gap2_cos_sum", "cnum_conflict", "cnum_c21", "n_tset", "a_tset", "fnum_eq", "num_conflict",
       "a_empty2", "state_cmp", "legal_1", "legal_2", "legal_jac", "key_n_s1", "n_ntok2", "a_ntok2", "b_n"]
out = {}
for split, feats in (("train", args.train_feats), ("test", args.test_feats)):
    files = sorted(glob.glob(os.path.join(CACHE, "feats", feats, "part_*.parquet")))
    cols = pl.scan_parquet(files).collect_schema().names()
    key = [k for k in KEY if k in cols]
    q = pl.scan_parquet(files).filter(pl.col("b_rank_cos_sum") == 1).select(["a", "b"] + key)
    d = q.collect()
    s1c = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s1.parquet"), columns=["country"])["country"].to_numpy()
    d = d.with_columns(pl.Series("country", s1c[d["a"].to_numpy()]))
    if split == "test" and args.test_scores and os.path.exists(os.path.join(CACHE, "feats", args.test_scores)):
        sc = pl.read_parquet(os.path.join(CACHE, "feats", args.test_scores))
        d = d.join(sc, on=["a", "b"], how="left")
        key = key + ["p"]
    for c in sorted(set(d["country"].to_list())):
        x = d.filter(pl.col("country") == c)
        out[f"{split}_{c}"] = {"n": x.height, **{k: [round(float(v), 3) for v in
                               (x[k].mean(), x[k].quantile(0.1), x[k].quantile(0.5), x[k].quantile(0.9))] for k in key}}
        print(split, c, json.dumps(out[f"{split}_{c}"]), flush=True)
os.makedirs(os.path.join(RESULTS, "eda"), exist_ok=True)
json.dump(out, open(os.path.join(RESULTS, "eda", f"shift_check_{args.test_feats}_{NORM}.json"), "w"), indent=1)
