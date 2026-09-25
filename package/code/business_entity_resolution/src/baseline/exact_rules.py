"""Baseline B0: deterministic exact-key matching on normalized fields.

Rules (each S2/S3 record is assigned to at most one S1, because the ground truth says so):
  R1  same country + same sorted name key
  R2  R1 + address agreement (shared house number or address token overlap >= 0.5)
  R3  same country + same sorted name core + same first number
  R4  R2 union (same country + identical address tokens + first number, name key token overlap >= 0.5)
A key that points to several S1 entities is ambiguous and is skipped.
Scored with macro F0.5 on the validation fold (fold 0); search space is all training S2/S3 records.
"""
import json
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402

OUT = os.path.join(RESULTS, "baseline")
os.makedirs(OUT, exist_ok=True)
gpu_init()
t0 = time.time()


def sorted_tokens(col):
    return pl.col(col).str.split(" ").list.sort().list.join(" ")


def load(k):
    d = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"),
                        columns=["id", "country", "core", "key", "am", "first_num", "nums"])
    return d.with_columns(sorted_tokens("key").alias("skey"), sorted_tokens("core").alias("score_"))


s1 = load(1)
rr = pl.concat([load(2), load(3)])
pairs, per = load_gt()
per = per.with_columns(pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
val_ids = per.filter(pl.col("fold") == VAL_FOLD)["s1"].to_list()
truth = {s: m for s, m in per.filter(pl.col("fold") == VAL_FOLD).select(["s1", "matches"]).iter_rows()}


def unique_join(left, right, keys):
    """join records to S1 on keys, drop keys that map to more than one S1."""
    r = right.group_by(keys).agg(pl.col("id").alias("s1s")).filter(pl.col("s1s").list.len() == 1) \
        .with_columns(pl.col("s1s").list.first().alias("s1")).drop("s1s")
    return left.join(r, on=keys, how="inner")


def addr_ok(df):
    a1 = pl.col("am").str.split(" ")
    a2 = pl.col("am_s1").str.split(" ")
    inter = a1.list.set_intersection(a2).list.len()
    small = pl.min_horizontal(a1.list.len(), a2.list.len())
    return ((pl.col("first_num") != "") & (pl.col("first_num") == pl.col("first_num_s1"))) | \
           ((small > 0) & (inter / small >= 0.5))


def score(pred_pairs, name):
    pp = pred_pairs.unique(subset=["id"], keep="first")        # one S1 per record
    pred = {}
    for s, r in pp.select(["s1", "id"]).iter_rows():
        pred.setdefault(s, []).append(r)
    m, _ = evaluate(pred, truth, val_ids)
    m["rule"] = name
    print(json.dumps(m), flush=True)
    return m


res = []
s1j = s1.rename({c: f"{c}_s1" for c in ["core", "key", "am", "first_num", "nums", "score_"]})
with Timer("R1"):
    r1 = unique_join(rr, s1.select(["id", "country", "skey"]).rename({"id": "id"}), ["country", "skey"])
    r1 = r1.join(s1j.select(["id", "am_s1", "first_num_s1"]).rename({"id": "s1"}), on="s1")
    res.append(score(r1, "R1 name key"))
with Timer("R2"):
    r2 = r1.filter(addr_ok(r1))
    res.append(score(r2, "R2 name key + address agreement"))
with Timer("R3"):
    r3 = unique_join(rr.filter(pl.col("first_num") != ""), s1.select(["id", "country", "score_", "first_num"]),
                     ["country", "score_", "first_num"])
    res.append(score(r3, "R3 name core + first number"))
with Timer("R4"):
    r4a = unique_join(rr.filter(pl.col("am") != ""), s1.select(["id", "country", "am", "first_num"]),
                      ["country", "am", "first_num"])
    r4a = r4a.join(s1j.select(["id", "skey"]).rename({"id": "s1", "skey": "skey_s1"}), on="s1")
    k1, k2 = pl.col("skey").str.split(" "), pl.col("skey_s1").str.split(" ")
    r4a = r4a.filter(k1.list.set_intersection(k2).list.len() / pl.min_horizontal(k1.list.len(), k2.list.len()) >= 0.5)
    r4 = pl.concat([r2.select(["id", "s1"]), r4a.select(["id", "s1"])])
    res.append(score(r4, "R4 R2 + (same address, name overlap)"))
json.dump({"results": res, "runtime_s": time.time() - t0}, open(os.path.join(OUT, "exact_rules_val.json"), "w"), indent=1)
