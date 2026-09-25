"""Are the confident false positives (records of no S1 that look like an S1) separable from true pairs?
Among pairs with oof p >= 0.9 that are the record's best S1, compare error rates across simple descriptors."""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_source, NORM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="xgb_c1")
ap.add_argument("--feats", default="c1_train")
args = ap.parse_args()
gpu_init()
rdir = os.path.join(RESULTS, "xgboost", args.tag)
D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))
D = D.with_columns(pl.col("p").rank("ordinal", descending=True).over("b").alias("rk")).filter((pl.col("rk") == 1) & (pl.col("p") >= 0.9))
files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
F = pl.scan_parquet(files).select(["a", "b", "b_has_match", "legal_1", "legal_2", "legal_jac", "fnum_eq", "num_conflict",
                                   "k_eq", "a_tset", "n_tset", "src3", "a_empty2", "nonlatin", "is_domain",
                                   "key_n_s1", "key_n_r"]).join(D.lazy().select(["a", "b", "y", "p"]), on=["a", "b"]).collect()
N1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=["legal", "core", "first_num", "nums"])
NR = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=["legal", "core", "first_num", "nums"])
                for k in (2, 3)])
a, b = F["a"].to_numpy(), F["b"].to_numpy()
F = F.with_columns(pl.Series("leg1", N1["legal"].to_numpy()[a]), pl.Series("leg2", NR["legal"].to_numpy()[b]),
                   pl.Series("fn1", N1["first_num"].to_numpy()[a]), pl.Series("fn2", NR["first_num"].to_numpy()[b]))
# how many confident records the S1 has in total
conf = D.group_by("a").len().rename({"len": "a_conf"})
F = F.join(conf, on="a", how="left")
F = F.with_columns(err=(pl.col("y") == 0).cast(pl.Int32),
                   leg_pat=pl.when((pl.col("leg1") == "") & (pl.col("leg2") == "")).then(pl.lit("both none"))
                   .when(pl.col("leg1") == pl.col("leg2")).then(pl.lit("same"))
                   .when(pl.col("leg1") == "").then(pl.lit("S1 none, rec has"))
                   .when(pl.col("leg2") == "").then(pl.lit("S1 has, rec none"))
                   .otherwise(pl.lit("different")),
                   num_pat=pl.when((pl.col("fn1") == "") | (pl.col("fn2") == "")).then(pl.lit("missing"))
                   .when(pl.col("fn1") == pl.col("fn2")).then(pl.lit("equal"))
                   .when(pl.col("fn1").str.contains(pl.col("fn2"), literal=True) | pl.col("fn2").str.contains(pl.col("fn1"), literal=True))
                   .then(pl.lit("one contains other")).otherwise(pl.lit("different")))
out = {"n_pairs_p>=0.9_best": F.height, "error_rate": float(F["err"].mean()),
       "fp_distractor_share": float(F.filter(pl.col("err") == 1)["b_has_match"].eq(0).mean())}
for col in ("leg_pat", "num_pat", "src3", "a_empty2", "nonlatin", "is_domain", "k_eq", "a_conf"):
    t = F.group_by(col).agg(pl.len().alias("n"), pl.col("err").mean().alias("err_rate")).sort("n", descending=True).head(15)
    out[col] = t.to_dicts()
# legal pattern x number pattern
t = F.group_by(["leg_pat", "num_pat"]).agg(pl.len().alias("n"), pl.col("err").mean().alias("err_rate")).sort("n", descending=True)
out["leg_x_num"] = t.to_dicts()
# top legal transitions among errors vs correct
for nm, sub in (("err", F.filter(pl.col("err") == 1)), ("ok", F.filter(pl.col("err") == 0))):
    out[f"legal_transitions_{nm}"] = sub.group_by(["leg1", "leg2"]).len().sort("len", descending=True).head(25).to_dicts()
print(json.dumps(out, indent=1, default=str), flush=True)
json.dump(out, open(os.path.join(rdir, "confuser_mining.json"), "w"), indent=1, default=str)
