"""Simulate the test universe per country on the labelled training data (generalises simulate_drop.py / SIM19).

Why: SIM19 matched the test's records-per-S1 (40 % distractors) by dropping 19 % of S1, but not the test's DENSITY.
Test US has 0.66 M S1 against 1.32 M in train (50 %); India 0.81 M vs 0.88 M (92 %); France 0.26 M. With half as many
S1 in the index, every record's runner-up is weaker, competition features and count features (b_n, a_n, key_n_s1,
key_n_r) shift, and models trained on the dense universe over-accept on test US (phase-2 ablation: US predicted matches
per S1 are 1-9 % higher on test than on validation, India within 0.2 %; the leaderboard order follows it).

Per country c, with keep_c and drop_c:
  1. keep a random share keep_c of the entities: S1 (hash of id) with all their true records, and the same share of
     the distractor records (hash of record id); everything else leaves the universe entirely
  2. drop a share drop_c of the kept S1 (the SIM19 hash): their records stay as distractors
  3. recompute everything that depends on the universe: retrieval ranks, competition features, b_n, a_n, key_n_s1,
     key_n_r, b_has_match
The default plan reproduces test US (0.62 x 0.81 x 1.32 M = 0.66 M S1, 5.8 records per S1) and SIM19 for India.
Output: feats/<src>_<tag>/part_*.parquet and drop_<code>.npy (S1 absent from the universe: never scored).

  python code/v6/simulate_universe.py --src c2_train --plan US:0.62:0.19,India:1.0:0.19 --tag dmsA --code 62
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, NORM, RESULTS, load_gt  # noqa: E402


def h(ids, salt):
    return np.array([(zlib.crc32((x + salt).encode()) % 10000) / 10000 for x in ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="c2_train")
    ap.add_argument("--plan", default="US:0.62:0.19,India:1.0:0.19", help="country:keep:drop,...")
    ap.add_argument("--tag", default="dmsA")
    ap.add_argument("--code", type=int, default=62, help="drop_<code>.npy name (an int, for the --simdrop options)")
    args = ap.parse_args()
    t0 = time.time()
    plan = {p.split(":")[0]: (float(p.split(":")[1]), float(p.split(":")[2])) for p in args.plan.split(",")}
    s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=["id", "country", "key"])
    rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=["id", "country", "key"]) for k in (2, 3)])
    c1, cr = s1["country"].to_numpy(), rr["country"].to_numpy()
    keep1 = np.array([plan.get(c, (1.0, 0.0))[0] for c in c1])
    drop1 = np.array([plan.get(c, (1.0, 0.0))[1] for c in c1])
    hk1 = h(s1["id"].to_list(), "|keep")
    kept_s1 = hk1 < keep1
    dropped = kept_s1 & (h(s1["id"].to_list(), "|drop") < drop1)
    present_s1 = kept_s1 & ~dropped
    # records: true records follow their S1's keep, distractors are kept with the same share
    pairs, _ = load_gt()
    m1 = pl.DataFrame({"s1": s1["id"], "a": np.arange(s1.height)})
    mr = pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height)})
    T = pairs.join(m1, on="s1").join(mr, on="rid")
    true_a = np.full(rr.height, -1, np.int64)
    true_a[T["b"].to_numpy()] = T["a"].to_numpy()
    keepr_share = np.array([plan.get(c, (1.0, 0.0))[0] for c in cr])
    kept_r = np.where(true_a >= 0, kept_s1[np.maximum(true_a, 0)], h(rr["id"].to_list(), "|keep") < keepr_share)
    absent = ~present_s1
    np.save(os.path.join(CACHE, f"drop_{args.code}.npy"), absent)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.src, "part_*.parquet")))
    cols = pl.scan_parquet(files).collect_schema().names()
    rank_cols = [c for c in cols if c.startswith("r_")]
    cos_cols = [c for c in cols if c.startswith("cos_") and c not in ("cos_name", "cos_addr")]
    ctx_cols = ["cos_sum"] + [c for c in cos_cols if c != "cos_sum"]
    derived = {"b_n", "a_n", "key_n_s1", "key_n_r"} | {f"{p}_{c}" for c in ctx_cols for p in ("b_gap", "b_rank", "a_gap", "a_rank", "b_gap2")}
    derived = [c for c in cols if c in derived]
    out_dir = os.path.join(CACHE, "feats", f"{args.src}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    with Timer("slim frame + filter to the simulated universe"):
        S = pl.scan_parquet(files).select(["a", "b", "y"] + rank_cols + ctx_cols).collect()
        n_before = S.height
        S = S.filter(pl.Series(present_s1[S["a"].to_numpy()] & kept_r[S["b"].to_numpy()]))
    with Timer("recompute ranks, competition and count features"):
        ex = [pl.when(pl.col(c) < 99).then(pl.col(c).rank("ordinal").over("b") - 1).otherwise(99).cast(pl.Int8).alias(c) for c in rank_cols]
        S = S.with_columns(ex)
        ex = [pl.len().over("b").cast(pl.Int32).alias("b_n"), pl.len().over("a").cast(pl.Int32).alias("a_n")]
        for c in ctx_cols:
            ex += [(pl.col(c) - pl.col(c).max().over("b")).alias(f"b_gap_{c}"),
                   pl.col(c).rank("ordinal", descending=True).over("b").cast(pl.Int32).alias(f"b_rank_{c}"),
                   (pl.col(c) - pl.col(c).max().over("a")).alias(f"a_gap_{c}"),
                   pl.col(c).rank("ordinal", descending=True).over("a").cast(pl.Int32).alias(f"a_rank_{c}")]
        S = S.with_columns(ex)
        for c in ctx_cols:
            sec = S.filter(pl.col(f"b_rank_{c}") == 2).select(["b", pl.col(c).alias("_sec")])
            S = S.join(sec, on="b", how="left").with_columns((pl.col(c) - pl.col("_sec").fill_null(0.0)).alias(f"b_gap2_{c}")).drop("_sec")
        # corpus counts inside the simulated universe (key frequency among present S1 / kept records of the country)
        k1 = s1.filter(pl.Series(present_s1)).group_by(["country", "key"]).len().rename({"len": "key_n_s1"})
        kr = rr.filter(pl.Series(kept_r)).group_by(["country", "key"]).len().rename({"len": "key_n_r"})
        ka = s1.select(["country", "key"]).with_row_index("a").join(k1, on=["country", "key"], how="left").select(
            [pl.col("a").cast(S.schema["a"]), pl.col("key_n_s1").fill_null(0).cast(pl.Int32)])
        kb = rr.select(["country", "key"]).with_row_index("b").join(kr, on=["country", "key"], how="left").select(
            [pl.col("b").cast(S.schema["b"]), pl.col("key_n_r").fill_null(0).cast(pl.Int32)])
        S = S.join(ka, on="a", how="left").join(kb, on="b", how="left")
        has = S.filter(pl.col("y") == 1).select("b").unique().with_columns(pl.lit(1, pl.Int8).alias("b_has_match"))
        S = S.join(has, on="b", how="left").with_columns(pl.col("b_has_match").fill_null(0)).sort(["b", "a"])
    with Timer("write parts"):
        off = 0
        replace = rank_cols + derived + ["b_has_match"]
        for i, f in enumerate(files):
            d = pl.read_parquet(f)
            d = d.filter(pl.Series(present_s1[d["a"].to_numpy()] & kept_r[d["b"].to_numpy()]))
            new = S[off:off + d.height]
            assert (new["a"].to_numpy() == d["a"].to_numpy()).all() and (new["b"].to_numpy() == d["b"].to_numpy()).all()
            off += d.height
            d = d.with_columns([new[c].alias(c) for c in replace if c in new.columns and c in d.columns])
            d.write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        assert off == S.height
    rep = {"plan": plan, "pairs_before": n_before, "pairs_after": S.height, "code": args.code, "runtime_s": time.time() - t0, "by_country": {}}
    for c in sorted(set(c1.tolist())):
        m = c1 == c
        nr = int((kept_r & (cr == c)).sum())
        rep["by_country"][c] = {"s1_present": int((present_s1 & m).sum()), "records": nr,
                                "records_per_s1": nr / max(int((present_s1 & m).sum()), 1),
                                "distractor_share": float(1 - (kept_r & (cr == c) & (true_a >= 0) &
                                                               present_s1[np.maximum(true_a, 0)]).sum() / max(nr, 1))}
    os.makedirs(os.path.join(RESULTS, "v6"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v6", f"universe_{args.src}_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
