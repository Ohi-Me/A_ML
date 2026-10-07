"""The bi-encoder cosine feature is computed differently on validation and on test.

  validation / training : cos_e3 of pair (a, b) = cosine under the ONE fold model that never saw a's fold
                          (model_f, f = fold_of(S1 id)); competition features (b_gap / b_rank / a_gap / a_rank / b_gap2)
                          therefore compare cosines of different models whenever the competing S1 sit in other folds.
  test (V2 and V3)      : cos_e3 = MEAN of the five fold models' cosines (oof_biencoder.py --split test), and the
                          competition features compare these averaged, lower-noise cosines.

Steps
  cos       all five fold-model cosines for every test candidate pair  -> emb/e3_test/paircos5_c2.parquet
            (sanity: their mean must equal the cos_e3 that V2 / V3 used)
  variants  cos_e3 + competition features for each variant -> p2/cosvar/c2_test_<var>.parquet
              mean5 : what V2 / V3 used (sanity check against the feature parts)
              hash1 : validation logic on test (model fold_of(S1 id), same hash as the training folds)
  report    distribution shift / correlation / noise between the variants, against the validation fold
  stage1    re-score the test stage-1 blend with a cosine variant, for the refit models (V2 / V3) and the two half
            models (chain A / B)                                  -> feats/p2_test_s1_<models>_<var>.parquet

  python code/phase2/cos_variants.py cos
  python code/phase2/cos_variants.py variants
  python code/phase2/cos_variants.py report
  python code/phase2/cos_variants.py stage1 --var hash1 --models refit,halfA,halfB
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import ids  # noqa: E402
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from phase2.p2lib import COS_CTX, apply_override, ctx_features, device, dump, s1_hash_fold  # noqa: E402

PC5 = os.path.join(CACHE, "emb", "e3_test", "paircos5_c2.parquet")
VDIR = os.path.join(CACHE, "p2", "cosvar")
BLEND = "xgb_c2_blend_v2"
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def cmd_cos(args):
    import scipy.sparse as sp
    from embeddings.ngram_biencoder import GCSR, Encoder, encode_rows
    dev = device()
    bdir = os.path.join(CACHE, "blocking", f"{args.btag}_test")
    name = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
    addr = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
    hn, ha = name.shape[1], addr.shape[1]
    HAS = torch.from_numpy((np.diff(addr.indptr) > 0).astype(np.float32)).to(dev)
    NM, AD = GCSR(name, dev), GCSR(addr, dev)
    n_all = name.shape[0]
    del name, addr
    n1 = ids("test")[0].height
    C = pl.read_parquet(os.path.join(CACHE, "cands", "c2_test.parquet"), columns=["a", "b"])
    out = C
    for f in range(5):
        with Timer(f"fold model {f}: test pair cosines"):
            m = Encoder(hn, ha).to(dev)
            m.load_state_dict(torch.load(os.path.join(CACHE, "emb", "models", f"e3_fold{f}.pt"), map_location=dev))
            E = encode_rows(m, NM, AD, HAS, torch.arange(n_all, device=dev))
            del m
            a = torch.from_numpy(C["a"].to_numpy().astype(np.int64)).to(dev)
            b = torch.from_numpy(C["b"].to_numpy().astype(np.int64) + n1).to(dev)
            c = np.zeros(C.height, np.float32)
            for s in range(0, C.height, 4_000_000):
                c[s:s + 4_000_000] = (E[a[s:s + 4_000_000]].float() * E[b[s:s + 4_000_000]].float()).sum(1).cpu().numpy()
            out = out.with_columns(pl.Series(f"c{f}", c))
            del E, a, b
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    out.write_parquet(PC5)
    old = pl.read_parquet(os.path.join(CACHE, "emb", "e3_test", "paircos_c2.parquet"))
    j = out.join(old, on=["a", "b"], how="left")
    mean5 = np.mean(np.stack([j[f"c{f}"].to_numpy() for f in range(5)]), 0)
    rep = {"pairs": out.height, "max_abs_diff_mean5_vs_v3_cos_e3": float(np.nanmax(np.abs(mean5 - j["cos_e3"].to_numpy())))}
    print(json.dumps(rep), flush=True)
    dump(rep, "cos", "cos_step.json")


def variant_cos(P, var, hfold):
    c = np.stack([P[f"c{f}"].to_numpy() for f in range(5)], 1)
    if var == "mean5":
        return c.mean(1)
    if var == "hash1":
        return c[np.arange(len(c)), hfold[P["a"].to_numpy()]]
    if var.startswith("fold"):                        # one fixed model for everything (sensitivity check)
        return c[:, int(var[4:])]
    raise ValueError(var)


def cmd_variants(args):
    os.makedirs(VDIR, exist_ok=True)
    P = pl.read_parquet(PC5)
    hfold = s1_hash_fold("test")
    rep = {}
    for var in args.vars.split(","):
        with Timer(f"variant {var}: cosine + competition features"):
            C = P.select(["a", "b"]).with_columns(pl.Series("cos_e3", variant_cos(P, var, hfold).astype(np.float32)))
            C = ctx_features(C, "cos_e3").select(["a", "b"] + COS_CTX)
            C.write_parquet(os.path.join(VDIR, f"c2_test_{var}.parquet"))
        if var == "mean5":
            # the rebuilt mean5 features must equal what the V2 / V3 test feature parts contain
            f0 = sorted(glob.glob(os.path.join(CACHE, "feats", "c2_test", "part_*.parquet")))[0]
            d = pl.read_parquet(f0, columns=["a", "b"] + COS_CTX)
            j = d.join(C, on=["a", "b"], how="left", suffix="_new")
            rep["mean5_check_part0"] = {c: float(np.nanmax(np.abs(j[c].cast(pl.Float64).to_numpy() -
                                                                   j[c + "_new"].cast(pl.Float64).to_numpy())))
                                        for c in COS_CTX}
            print("mean5 sanity (max |diff| vs feature part 0):", rep["mean5_check_part0"], flush=True)
    dump(rep, "cos", "variants_step.json")


def qstats(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            **{f"q{int(q * 100):02d}": float(v) for q, v in zip(QS, np.quantile(x, QS))}}


def ks(x, y, n=500_000, seed=0):
    from scipy.stats import ks_2samp
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(n, len(x)), replace=False)] if len(x) else x
    y = y[rng.choice(len(y), min(n, len(y)), replace=False)] if len(y) else y
    if len(x) < 10 or len(y) < 10:
        return None
    return float(ks_2samp(x, y).statistic)


def cmd_report(args):
    from scipy.stats import pearsonr, spearmanr
    t0 = time.time()
    rep = {"definition": {"val": "c2_train_sim19 parts, fold 0 (single held-out fold model, as trained)",
                          "test_mean5": "what V2 / V3 fed their models", "test_hash1": "validation logic on test"}}
    cols = ["cos_e3", "b_gap_cos_e3", "b_gap2_cos_e3", "a_gap_cos_e3", "b_rank_cos_e3"]
    # validation reference (fold 0) with the record's best stage-1 pair flagged
    files = sorted(glob.glob(os.path.join(CACHE, "feats", "c2_train_sim19", "part_*.parquet")))
    V = pl.concat([pl.read_parquet(f, columns=["a", "b", "fold", "y"] + cols).filter(pl.col("fold") == 0) for f in files])
    oof = pl.read_parquet(os.path.join(CACHE, "feats", "oof_xgb_c2_blend_v2.parquet"), columns=["a", "b", "p"])
    V = V.join(oof, on=["a", "b"], how="left")
    V = V.with_columns((pl.col("p") == pl.col("p").max().over("b")).alias("top1"))
    ctr = ids("train")[0]["country"].to_numpy()
    cte = ids("test")[0]["country"].to_numpy()
    V = V.with_columns(pl.Series("country", ctr[V["a"].to_numpy()]))
    T = {}
    s1t = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_final_v2.parquet"), columns=["a", "b", "p"])
    for var in ("mean5", "hash1"):
        d = pl.read_parquet(os.path.join(VDIR, f"c2_test_{var}.parquet")).select(["a", "b"] + cols)
        d = d.join(s1t, on=["a", "b"], how="left")
        d = d.with_columns((pl.col("p") == pl.col("p").max().over("b")).alias("top1"),
                           pl.Series("country", cte[d["a"].to_numpy()]))
        T[var] = d
    rep["distributions"] = {}
    for c in sorted(set(cte.tolist())):
        r = {}
        for sub in ("all", "top1"):
            fv = V.filter(pl.col("country") == c)
            if sub == "top1":
                fv = fv.filter(pl.col("top1"))
            for col in cols:
                x = {}
                if fv.height:
                    x["val"] = qstats(fv[col].to_numpy())
                    x["val_pos"] = qstats(fv.filter(pl.col("y") == 1)[col].to_numpy())
                    x["val_neg"] = qstats(fv.filter(pl.col("y") == 0)[col].to_numpy())
                for var, d in T.items():
                    fd = d.filter(pl.col("country") == c)
                    if sub == "top1":
                        fd = fd.filter(pl.col("top1"))
                    x[f"test_{var}"] = qstats(fd[col].to_numpy())
                    if fv.height:
                        x[f"ks_val_vs_test_{var}"] = ks(fv[col].to_numpy().astype(np.float64), fd[col].to_numpy().astype(np.float64))
                r[f"{sub}:{col}"] = x
        rep["distributions"][c] = r
    # agreement between the two computations on the same test pairs
    m5, h1 = T["mean5"], T["hash1"]
    assert (m5["a"].to_numpy() == h1["a"].to_numpy()).all() and (m5["b"].to_numpy() == h1["b"].to_numpy()).all()
    P = pl.read_parquet(PC5)
    spread = np.stack([P[f"c{f}"].to_numpy() for f in range(5)], 1).std(1)
    rep["agreement"] = {}
    for c in sorted(set(cte.tolist())):
        mk = (m5["country"] == c).to_numpy()
        for sub in ("all", "top1"):
            mm = mk & (m5["top1"].to_numpy() if sub == "top1" else True)
            x, y = m5["cos_e3"].to_numpy()[mm], h1["cos_e3"].to_numpy()[mm]
            rs = np.random.default_rng(0).choice(len(x), min(len(x), 2_000_000), replace=False)
            g5, g1 = m5["b_gap2_cos_e3"].to_numpy()[mm], h1["b_gap2_cos_e3"].to_numpy()[mm]
            rep["agreement"][f"{c}:{sub}"] = {
                "pearson_cos": float(pearsonr(x[rs], y[rs])[0]), "spearman_cos": float(spearmanr(x[rs], y[rs])[0]),
                "mean_abs_diff_cos": float(np.abs(x - y).mean()), "std_ratio_hash1_over_mean5": float(y.std() / max(x.std(), 1e-9)),
                "pearson_b_gap2": float(pearsonr(g5[rs], g1[rs])[0]), "mean_b_gap2_mean5": float(g5.mean()),
                "mean_b_gap2_hash1": float(g1.mean()),
                "mean_between_model_std": float(spread[mm].mean()),
                "record_top_cos_s1_changes": None}
    # does the S1 with the highest cosine for a record change between the two computations?
    for c in sorted(set(cte.tolist())):
        mk = pl.Series((m5["country"] == c).to_numpy())
        r1 = m5.filter(mk).filter(pl.col("b_rank_cos_e3") == 1).select(["b", "a"])
        r2 = h1.filter(mk).filter(pl.col("b_rank_cos_e3") == 1).select(["b", pl.col("a").alias("a2")])
        j = r1.join(r2, on="b")
        rep["agreement"][f"{c}:all"]["record_top_cos_s1_changes"] = float((j["a"] != j["a2"]).mean())
    rep["runtime_s"] = time.time() - t0
    dump(rep, "cos", "cos_report.json")


def cmd_stage1(args):
    import xgboost as xgb
    bl = json.load(open(os.path.join(RESULTS, "xgboost", BLEND, "report.json")))["blend"]
    reps = {t: json.load(open(os.path.join(RESULTS, "xgboost", t, "report.json"))) for t in (bl["full"], bl["noemb"])}
    seen = set(ids("train")[0]["country"].unique().to_list())
    cty = ids("test")[0]["country"].to_numpy()
    w = np.where(np.isin(cty, list(seen)), bl["w_seen"], bl["w_unseen"]).astype(np.float32)
    ov = pl.read_parquet(os.path.join(VDIR, f"c2_test_{args.var}.parquet")) if args.var != "none" else None
    files = sorted(glob.glob(os.path.join(CACHE, "feats", "c2_test", "part_*.parquet")))
    for kind in args.models.split(","):
        out_p = os.path.join(CACHE, "feats", f"p2_test_s1_{kind}_{args.var}.parquet")
        if os.path.exists(out_p) and not args.force:
            print("exists", out_p, flush=True)
            continue
        mods = {}
        for t in (bl["full"], bl["noemb"]):
            if kind == "refit":
                mp = os.path.join(RESULTS, "final", "final_v2", f"stage1_{t}_all_folds.json")
            else:
                mp = os.path.join(RESULTS, "xgboost", t, f"model_{kind[-1]}.json")
            mods[t] = xgb.Booster(model_file=mp)
        with Timer(f"stage-1 {kind} with cosine variant {args.var}"):
            outs = []
            for f in files:
                d = pl.read_parquet(f).rename({"r_e2f_emb": "r_e3_emb"}, strict=False)
                d = apply_override(d, ov)
                ps = [mods[t].predict(xgb.DMatrix(d.select([pl.col(c).cast(pl.Float32) for c in reps[t]["features"]]).to_numpy()))
                      for t in (bl["full"], bl["noemb"])]
                ww = w[d["a"].to_numpy()]
                outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", (ww * ps[0] + (1 - ww) * ps[1]).astype(np.float32))))
            T = pl.concat(outs)
            T.write_parquet(out_p)
        if kind == "refit" and args.var == "mean5":
            ref = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_final_v2.parquet"))
            dump({"max_abs_diff_vs_test_stage1_final_v2": float(np.abs(ref["p"].to_numpy() - T["p"].to_numpy()).max())},
                 "cos", "stage1_refit_mean5_check.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cos", "variants", "report", "stage1"])
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--vars", default="mean5,hash1")
    ap.add_argument("--var", default="hash1")
    ap.add_argument("--models", default="refit,halfA,halfB")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"cos": cmd_cos, "variants": cmd_variants, "report": cmd_report, "stage1": cmd_stage1}[args.step](args)


if __name__ == "__main__":
    main()
