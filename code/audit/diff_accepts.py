"""Which pairs does V3 accept that V2 rejects, on test vs on validation (where we know if they are right)?

On validation V3 adds ~0.01 matches per US S1 over V2, on test ~0.07 (India: ~0.02 on both). This profiles the
V3-only accepts per country: stage-1 score, cross-encoder coverage and score, GNN score, key pair features, and
prints raw examples of the test ones.

  python code/audit/diff_accepts.py
"""
import glob
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from audit.gap_audit import OUT, decode, iso_from  # noqa: E402
from common.decide import Scorer, ids  # noqa: E402
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, load_source  # noqa: E402

COLS = ["cos_sum", "cos_e3", "cos_name", "cos_addr", "n_tset", "a_tset", "k_eq", "state_cmp", "fnum_eq", "fnum_lev",
        "cnum_conflict", "num_conflict", "a_empty2", "legal_fam_cmp", "nonlatin", "is_domain", "b_n", "a_n", "key_n_s1",
        "b_gap2_cos_sum", "b_rank_cos_sum"]


def decisions(split):
    v2o, v3o = "oof_xgb_c2_blend_s2_v2.parquet", "oof_gnn_mlx_v2.parquet"
    if split == "train":
        T2 = pl.read_parquet(os.path.join(CACHE, "feats", v2o)).select(["a", "b", "p"])
        T3 = pl.read_parquet(os.path.join(CACHE, "feats", v3o)).select(["a", "b", "p"])
    else:
        T2 = pl.read_parquet(os.path.join(CACHE, "feats", "test_scores_final_v2.parquet")).select(["a", "b", "p"])
        T3 = pl.read_parquet(os.path.join(CACHE, "gnn", "gnn_mlx_v2_test_scores.parquet")).select(
            ["a", "b", (1 / (1 + (-pl.col("gnn")).exp())).cast(pl.Float32).alias("p")])
    k2, _ = decode(T2, iso_from(v2o))
    k3, _ = decode(T3, iso_from(v3o))
    only3 = k3.join(k2.select(["a", "b"]), on=["a", "b"], how="anti").rename({"p": "p3cal"})
    only2 = k2.join(k3.select(["a", "b"]), on=["a", "b"], how="anti").rename({"p": "p2cal"})
    return only3, only2


def attach(D, split):
    feats = "c2_train_sim19" if split == "train" else "c2_test"
    files = sorted(glob.glob(os.path.join(CACHE, "feats", feats, "part_*.parquet")))
    F = pl.concat([pl.read_parquet(f, columns=["a", "b"] + COLS).join(D.select(["a", "b"]), on=["a", "b"], how="semi")
                   for f in files])
    D = D.join(F, on=["a", "b"], how="left")
    x = pl.read_parquet(os.path.join(CACHE, "xenc", f"ml_hardv2_{split}_scores.parquet"))
    D = D.join(x, on=["a", "b"], how="left")
    s1 = pl.read_parquet(os.path.join(CACHE, "feats", "oof_xgb_c2_blend_v2.parquet" if split == "train"
                                      else "test_stage1_final_v2.parquet"), columns=["a", "b", "p"]).rename({"p": "p1"})
    return D.join(s1, on=["a", "b"], how="left")


def profile(D):
    out = {"n": D.height, "has_xenc": float(D["mlxenc"].is_not_null().mean()),
           "xenc_mean": float(D["mlxenc"].mean() or 0), "p1_median": float(D["p1"].median()),
           "p1_lt_0.5": float((D["p1"] < 0.5).mean())}
    for c in COLS:
        out[c] = float(D[c].cast(pl.Float64).mean())
    return out


def main():
    gpu_init()
    rep = {}
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    for split in ("train", "test"):
        with Timer(split):
            only3, only2 = decisions(split)
            cty = ids(split)[0]["country"].to_numpy()
            n1 = {c: int((cty == c).sum()) for c in set(cty.tolist())}
            if split == "train":
                val = sc.fold == 0
                only3 = only3.filter(pl.Series(val[only3["a"].to_numpy()]))
                only2 = only2.filter(pl.Series(val[only2["a"].to_numpy()]))
                n1 = {c: int((val & (cty == c)).sum()) for c in n1}
                only3 = only3.with_columns(pl.Series("y", (sc.true_s1_of_b[only3["b"].to_numpy()] == only3["a"].to_numpy()).astype(np.int8)))
                only2 = only2.with_columns(pl.Series("y", (sc.true_s1_of_b[only2["b"].to_numpy()] == only2["a"].to_numpy()).astype(np.int8)))
            only3 = attach(only3, split).with_columns(pl.Series("country", cty[only3["a"].to_numpy()]))
            r = {}
            for c in sorted(n1):
                d3 = only3.filter(pl.col("country") == c)
                pr = profile(d3)
                pr["per_s1"] = d3.height / max(n1[c], 1)
                pr["v2_only_per_s1"] = int((cty[only2["a"].to_numpy()] == c).sum()) / max(n1[c], 1)
                if "y" in d3.columns:
                    pr["precision_v3_only"] = float(d3["y"].mean())
                    d2 = only2.filter(pl.Series(cty[only2["a"].to_numpy()] == c))
                    pr["precision_v2_only"] = float(d2["y"].mean())
                r[c] = pr
            rep[split] = r
            print(split, json.dumps(r, indent=1), flush=True)
            if split == "test":
                s1 = load_source("test", 1)
                rr = pl.concat([load_source("test", 2), load_source("test", 3)])
                ex = only3.filter(pl.col("country") == "US").sample(n=min(40, only3.height), seed=1)
                print("\nexamples: V3-only accepts on US test (S1 || record | p1 xenc gnn-cal)")
                for row in ex.iter_rows(named=True):
                    a, b = row["a"], row["b"]
                    print(f"  {s1['business_name'][a]} | {s1['business_address'][a]}  ||  {rr['entity_id'][b]} "
                          f"{rr['business_name'][b]} | {rr['business_address'][b]}   | {row['p1']:.3f} "
                          f"{row['mlxenc'] if row['mlxenc'] is None else round(row['mlxenc'], 2)} {row['p3cal']:.3f}")
                only3.write_parquet(os.path.join(CACHE, "feats", "audit_v3only_test.parquet"))
    json.dump(rep, open(os.path.join(OUT, "diff_accepts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
