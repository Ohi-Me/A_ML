"""Make the validation look like the test set: drop a share of S1 entities so that their records become
distractors (records with no S1), then recompute every feature that depends on the candidate set.

Why: test has 5.76 S2/S3 records per S1, train 4.68, while predicted matches per S1 are the same (~3.35), so
about 40% of test records have no S1 against 26% in train. Distractors cause most false merges.
Dropping 19% of S1 gives 4.68 / 0.81 = 5.8 records per S1, like test.

  python code/xgboost/simulate_drop.py --feats c2_train --drop 0.19
Output: data/cache/feats/<feats>_sim<pct>/part_*.parquet and data/cache/drop_<pct>.npy (dropped S1 rows)."""
import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, NORM, RESULTS  # noqa: E402


def drop_mask(split, frac, norm=None):
    norm = norm or NORM
    ids = pl.read_parquet(os.path.join(CACHE, f"norm_{norm}_{split}_s1.parquet"), columns=["id"])["id"].to_list()
    return np.array([(zlib.crc32((x + "|drop").encode()) % 10000) < frac * 10000 for x in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--drop", type=float, default=0.19)
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    pct = int(round(args.drop * 100))
    out_dir = os.path.join(CACHE, "feats", f"{args.feats}_sim{pct}")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    cols = pl.scan_parquet(files).collect_schema().names()
    dm = drop_mask("train", args.drop)
    np.save(os.path.join(CACHE, f"drop_{pct}.npy"), dm)
    rank_cols = [c for c in cols if c.startswith("r_")]
    cos_cols = [c for c in cols if c.startswith("cos_") and c not in ("cos_name", "cos_addr")]
    ctx_cols = [c for c in cos_cols if c != "cos_sum" and not c.startswith("cos_sum")]
    ctx_cols = ["cos_sum"] + ctx_cols
    derived = {"b_n", "a_n"} | {f"{p}_{c}" for c in ctx_cols for p in ("b_gap", "b_rank", "a_gap", "a_rank", "b_gap2")}
    derived = [c for c in cols if c in derived]
    with Timer("slim frame + filter"):
        S = pl.scan_parquet(files).select(["a", "b", "y"] + rank_cols + ctx_cols).collect()
        n_before = S.height
        keep = ~dm[S["a"].to_numpy()]
        S = S.filter(pl.Series(keep))
    with Timer("recompute ranks and context"):
        # retrieval ranks among the S1 that are still there (rank 99 = not in that list stays 99; rows with 99 sort
        # last, so the ordinal rank of an in-list row is its rank inside the list)
        exprs = []
        for c in rank_cols:
            exprs.append(pl.when(pl.col(c) < 99).then(
                (pl.col(c).rank("ordinal").over("b") - 1)).otherwise(99).cast(pl.Int8).alias(c))
        S = S.with_columns(exprs)
        ex = [pl.len().over("b").cast(pl.Int32).alias("b_n"), pl.len().over("a").cast(pl.Int32).alias("a_n")]
        for c in ctx_cols:
            ex += [(pl.col(c) - pl.col(c).max().over("b")).alias(f"b_gap_{c}"),
                   pl.col(c).rank("ordinal", descending=True).over("b").cast(pl.Int32).alias(f"b_rank_{c}"),
                   (pl.col(c) - pl.col(c).max().over("a")).alias(f"a_gap_{c}"),
                   pl.col(c).rank("ordinal", descending=True).over("a").cast(pl.Int32).alias(f"a_rank_{c}")]
        S = S.with_columns(ex)
        for c in ctx_cols:
            sec = S.filter(pl.col(f"b_rank_{c}") == 2).select(["b", pl.col(c).alias("_sec")])
            S = S.join(sec, on="b", how="left").with_columns(
                (pl.col(c) - pl.col("_sec").fill_null(0.0)).alias(f"b_gap2_{c}")).drop("_sec")
        # restore the row order of the filtered parts
        S = S.sort(["b", "a"])
        has = S.filter(pl.col("y") == 1).select("b").unique().with_columns(pl.lit(1, pl.Int8).alias("b_has_match"))
        S = S.join(has, on="b", how="left").with_columns(pl.col("b_has_match").fill_null(0)).sort(["b", "a"])
    with Timer("write parts"):
        off = 0
        replace = rank_cols + derived + ["b_has_match"]
        for i, f in enumerate(files):
            d = pl.read_parquet(f)
            d = d.filter(pl.Series(~dm[d["a"].to_numpy()]))
            new = S[off:off + d.height]
            assert (new["a"].to_numpy() == d["a"].to_numpy()).all() and (new["b"].to_numpy() == d["b"].to_numpy()).all()
            off += d.height
            d = d.with_columns([new[c].alias(c) for c in replace if c in new.columns])
            d.write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        assert off == S.height
    rep = {"pairs_before": n_before, "pairs_after": S.height, "dropped_s1": int(dm.sum()), "drop": args.drop,
           "records_with_true_s1_after": int(S.filter(pl.col("y") == 1)["b"].n_unique()), "runtime_s": time.time() - t0}
    print(json.dumps(rep), flush=True)
    os.makedirs(os.path.join(RESULTS, "xgboost"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "xgboost", f"simulate_{args.feats}_sim{pct}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
