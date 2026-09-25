"""Where do the V3-only accepts come from? Trace pairs through every stage, on test and on validation fold 0.

Groups (per split, per country):
  V3_only   accepted by V3, not by V2 (on test ~106k, of which ~49k US)
  V2_only   accepted by V2, not by V3
  common    accepted by both (sampled)
  reject    the record's best pair under V3's assignment, accepted by neither (sampled)
Stages traced per pair:
  stage 1   s1_p (blend fed to V3: refit on test, OOF on validation), s1_p_halfA (test: half model A, the kind the
            downstream models were trained on; validation: identical to s1_p), s1_rank_in_record
  candidates r_e3_emb (test: e2f rank renamed, see audit), r_b1_comb, b_rank_cos_sum, b_n, a_n
  cosine    cos_e3 as fed (test: mean of 5 models), cos_e3_hash1 (test, validation logic), b_gap2_cos_e3
  XLM-R     mlxenc (null outside the band), in_band
  GNN       gnn_logit, gnn_p_cal (V3 isotonic)
  V2        v2_p_cal (V2 stage 2, calibrated)
  decoder   pos_in_s1 (rank of the record among the S1's assigned records), n_assigned_s1, k_chosen_v3
Divergence: for each stage variable, KS(validation group, test group) against KS(validation population, test
population) of all assigned records; a group-specific shift is where the two differ.

  python code/phase2/diff_accepts.py
"""
import glob
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.decode import assign  # noqa: E402
from common.io import CACHE  # noqa: E402
from phase2.p2lib import calibrate, cpath, device, dump, exists, iso_from_oof, load_scores, qstats_all  # noqa: E402

DDIR = os.path.join(CACHE, "p2", "decisions")
TRACE_FEATS = ["r_e3_emb", "r_b1_comb", "b_rank_cos_sum", "b_n", "a_n", "cos_e3", "b_gap2_cos_e3", "cos_sum", "b_gap2_cos_sum"]
VARS = ["s1_p", "s1_p_halfA", "s1_rank_in_record", "r_e3_emb", "r_b1_comb", "b_rank_cos_sum", "b_n", "a_n", "cos_e3",
        "cos_e3_hash1", "b_gap2_cos_e3", "b_gap2_cos_sum", "mlxenc", "in_band", "gnn_logit", "gnn_p_cal", "v2_p_cal",
        "pos_in_s1", "n_assigned_s1", "k_chosen_v3"]
NSAMP = 300_000


def ks(x, y):
    from scipy.stats import ks_2samp
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 20 or len(y) < 20:
        return None
    return float(ks_2samp(x, y).statistic)


def build(split, sc):
    """per assigned record under V3: all stage variables (one row per record's best S1 = V3's assignment)."""
    tr = split == "train"
    v3o = pl.read_parquet(cpath("feats/oof_gnn_mlx_v2.parquet"))
    v2o = pl.read_parquet(cpath("feats/oof_xgb_c2_blend_s2_v2.parquet"))
    iso3, iso2 = iso_from_oof(v3o), iso_from_oof(v2o)
    if tr:
        G = v3o.select(["a", "b", pl.col("p").alias("gnn_prob")])
        V2 = v2o.select(["a", "b", "p"])
        S1 = pl.read_parquet(cpath("feats/oof_xgb_c2_blend_v2.parquet")).select(["a", "b", pl.col("p").alias("s1_p")])
        S1 = S1.with_columns(pl.col("s1_p").alias("s1_p_halfA"))
        X = pl.read_parquet(cpath("xenc/ml_hardv2_train_scores.parquet")).select(["a", "b", "mlxenc"])
        fdir, ren = "c2_train_sim19", {}
    else:
        G = load_scores("gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True).rename({"p": "gnn_prob"})
        V2 = pl.read_parquet(cpath("feats/test_scores_final_v2.parquet")).select(["a", "b", "p"])
        S1 = pl.read_parquet(cpath("feats/test_stage1_final_v2.parquet")).select(["a", "b", pl.col("p").alias("s1_p")])
        if exists("feats/test_stage1_halfA_v2.parquet"):
            S1 = S1.join(pl.read_parquet(cpath("feats/test_stage1_halfA_v2.parquet")).select(["a", "b", pl.col("p").alias("s1_p_halfA")]),
                         on=["a", "b"], how="left")
        X = pl.read_parquet(cpath("xenc/ml_hardv2_test_scores.parquet")).select(["a", "b", "mlxenc"])
        fdir, ren = "c2_test", {"r_e2f_emb": "r_e3_emb"}
    # V3 assignment (records to their best S1 under the calibrated GNN) and V3 / V2 decisions
    Gc = calibrate(G.rename({"gnn_prob": "p"}), iso3)
    A = assign(Gc, floor=0.01).rename({"p": "gnn_p_cal"})
    A = A.sort(["a", "gnn_p_cal"], descending=[False, True]).with_columns(
        (pl.int_range(pl.len()).over("a") + 1).alias("pos_in_s1"), pl.len().over("a").alias("n_assigned_s1"))
    k3 = pl.read_parquet(os.path.join(DDIR, f"V3_{'val' if tr else 'test'}.parquet"))
    k2 = pl.read_parquet(os.path.join(DDIR, f"V2_{'val' if tr else 'test'}.parquet"))
    kc = k3.group_by("a").len().rename({"len": "k_chosen_v3"})
    A = A.join(kc, on="a", how="left").with_columns(pl.col("k_chosen_v3").fill_null(0))
    A = A.join(k3.with_columns(pl.lit(True).alias("acc3")), on=["a", "b"], how="left") \
         .join(k2.with_columns(pl.lit(True).alias("acc2")), on=["a", "b"], how="left") \
         .with_columns(pl.col("acc3").fill_null(False), pl.col("acc2").fill_null(False))
    if tr:
        A = A.filter(pl.Series(sc.fold[A["a"].to_numpy()] == 0))
    # V2-only pairs may not be V3's assignment: add them explicitly
    v2only = k2.join(k3, on=["a", "b"], how="anti")
    if tr:
        v2only = v2only.filter(pl.Series(sc.fold[v2only["a"].to_numpy()] == 0))
    extra = v2only.join(A.select(["a", "b"]), on=["a", "b"], how="anti").with_columns(
        pl.lit(False).alias("acc3"), pl.lit(True).alias("acc2"))
    rng = np.random.default_rng(0)
    grp = np.where(A["acc3"].to_numpy() & ~A["acc2"].to_numpy(), "V3_only",
                   np.where(A["acc3"].to_numpy() & A["acc2"].to_numpy(), "common",
                            np.where(~A["acc3"].to_numpy() & A["acc2"].to_numpy(), "V2_only", "reject")))
    A = A.with_columns(pl.Series("group", grp))
    keep = np.isin(grp, ["V3_only", "V2_only"]) | (rng.random(len(grp)) < NSAMP / max(len(grp), 1))
    popu = A.select(["a", "b"]).sample(n=min(NSAMP, A.height), seed=1).with_columns(pl.lit("population").alias("group"))
    P = pl.concat([A.filter(pl.Series(keep)).select(["a", "b", "group"]),
                   extra.select(["a", "b"]).with_columns(pl.lit("V2_only").alias("group")), popu])
    keys = P.select(["a", "b"]).unique()
    # join every stage
    parts = sorted(glob.glob(os.path.join(CACHE, "feats", fdir, "part_*.parquet")))
    F = pl.concat([pl.read_parquet(f).rename(ren, strict=False).select(["a", "b"] + TRACE_FEATS).join(keys, on=["a", "b"], how="semi")
                   for f in parts])
    s1r = S1.with_columns(pl.col("s1_p").rank("ordinal", descending=True).over("b").alias("s1_rank_in_record"))
    D = P.join(F, on=["a", "b"], how="left").join(s1r.join(keys, on=["a", "b"], how="semi"), on=["a", "b"], how="left") \
         .join(X, on=["a", "b"], how="left").join(G, on=["a", "b"], how="left") \
         .join(A.select(["a", "b", "gnn_p_cal", "pos_in_s1", "n_assigned_s1", "k_chosen_v3"]), on=["a", "b"], how="left")
    D = D.join(calibrate(V2, iso2).rename({"p": "v2_p_cal"}), on=["a", "b"], how="left")
    D = D.with_columns((pl.col("gnn_prob") / (1 - pl.col("gnn_prob"))).log().alias("gnn_logit"),
                       pl.col("mlxenc").is_not_null().cast(pl.Float32).alias("in_band"))
    if not tr and exists("p2/cosvar/c2_test_hash1.parquet"):
        H = pl.read_parquet(cpath("p2/cosvar/c2_test_hash1.parquet")).select(["a", "b", pl.col("cos_e3").alias("cos_e3_hash1")])
        D = D.join(H.join(keys, on=["a", "b"], how="semi"), on=["a", "b"], how="left")
    elif tr:
        D = D.with_columns(pl.col("cos_e3").alias("cos_e3_hash1"))
    if tr:
        D = D.with_columns(pl.Series("y", (sc.true_s1_of_b[D["b"].to_numpy()] == D["a"].to_numpy()).astype(np.int8)))
    return D


def summarize(D, country, tr):
    out = {}
    c = country[D["a"].to_numpy()]
    for g in ("V3_only", "V2_only", "common", "reject", "population"):
        for cc in ["ALL"] + sorted(set(c.tolist())):
            m = (D["group"] == g).to_numpy() & ((c == cc) if cc != "ALL" else True)
            if not m.any():
                continue
            d = D.filter(pl.Series(m))
            e = {"n": int(m.sum())}
            if tr:
                e["precision_true_pair"] = float(d["y"].mean())
            e["vars"] = {v: qstats_all(d[v].cast(pl.Float64).to_numpy()) for v in VARS if v in d.columns}
            out[f"{g}|{cc}"] = e
    return out


def main():
    device()
    t0 = time.time()
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    c_tr, c_te = ids("train")[0]["country"].to_numpy(), ids("test")[0]["country"].to_numpy()
    for need in ("V2_val", "V3_val", "V2_test", "V3_test"):
        assert os.path.exists(os.path.join(DDIR, f"{need}.parquet")), "run code/phase2/ablation.py --only V2,V3 first"
    Dv, Dt = build("train", sc), build("test", sc)
    os.makedirs(os.path.join(CACHE, "p2"), exist_ok=True)
    Dv.write_parquet(os.path.join(CACHE, "p2", "trace_val.parquet"))
    Dt.write_parquet(os.path.join(CACHE, "p2", "trace_test.parquet"))
    rep = {"val_f0": summarize(Dv, c_tr, True), "test": summarize(Dt, c_te, False)}
    div = {}
    cv, ct = c_tr[Dv["a"].to_numpy()], c_te[Dt["a"].to_numpy()]
    for cc in ("US", "India", "ALL"):
        mv = (cv == cc) if cc != "ALL" else np.ones(len(cv), bool)
        mt = (ct == cc) if cc != "ALL" else np.ones(len(ct), bool)
        e = {}
        for v in VARS:
            if v not in Dv.columns or v not in Dt.columns:
                continue
            row = {}
            for g in ("population", "V3_only", "common", "V2_only"):
                x = Dv.filter(pl.Series(mv & (Dv["group"] == g).to_numpy()))[v].cast(pl.Float64).to_numpy()
                y = Dt.filter(pl.Series(mt & (Dt["group"] == g).to_numpy()))[v].cast(pl.Float64).to_numpy()
                row[f"ks_{g}"] = ks(x, y)
            if row.get("ks_population") and row.get("ks_V3_only") is not None:
                row["group_specific_shift"] = row["ks_V3_only"] - row["ks_population"]
            e[v] = row
        div[cc] = dict(sorted(e.items(), key=lambda kv: -(kv[1].get("ks_V3_only") or 0)))
    rep["divergence_val_vs_test"] = div
    # share of the V3-only accepts per test country and how many would be decided by the band / position
    tv = Dt.filter(pl.col("group") == "V3_only")
    ctv = c_te[tv["a"].to_numpy()]
    rep["test_V3_only_profile"] = {cc: {"n": int((ctv == cc).sum()),
                                        "in_band_share": float(tv.filter(pl.Series(ctv == cc))["in_band"].mean() or 0),
                                        "pos_ge_4_share": float((tv.filter(pl.Series(ctv == cc))["pos_in_s1"] >= 4).mean() or 0)}
                                   for cc in sorted(set(ctv.tolist()))}
    rep["runtime_s"] = time.time() - t0
    dump(rep, "diff_accepts.json")


if __name__ == "__main__":
    main()
