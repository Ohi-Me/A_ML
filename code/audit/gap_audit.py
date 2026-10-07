"""Why did V3 (sim-val 0.9908) score 0.9828 on the leaderboard while V2 (sim-val 0.9870) scored 0.9833?

Label-free comparison of validation vs test, per country, stage by stage:
  1. data prior per country (records per S1, true matches per S1, distractor share)
  2. decisions: predicted matches per S1, empty S1, mean calibrated p of the accepted pairs (val vs test, V2 and V3)
  3. stage-1 input shift: the test stage 1 came from an all-folds refit (100% of the data, x1.25 rounds) while every
     downstream model (stage 2, GNN, cross-encoder band, calibration) was trained on half-model scores (40% of the
     data). Score the test pairs with the half models too and compare.
The half-model test tables are saved for the chain-consistent runs (feats/test_stage1_half{A,B}_v2.parquet).

  python code/audit/gap_audit.py
"""
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, NORM, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "audit")
BAND = (0.02, 0.98)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def record_country(split):
    return pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s{k}.parquet"), columns=["country"])
                      for k in (2, 3)])["country"].to_numpy()


def iso_from(oof_name):
    oof = pl.read_parquet(os.path.join(CACHE, "feats", oof_name))
    tr = oof.filter(pl.col("fold").is_in([1, 2]))
    tr = tr.sample(n=min(8_000_000, tr.height), seed=0)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(tr["p"].to_numpy(), tr["y"].to_numpy())


def decode(T, iso):
    T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    A = assign(T, floor=0.01)
    keep, _ = expected_f_decode(A, gamma=1.0, M=2048)
    return keep.join(A.select(["a", "b", "p"]), on=["a", "b"], how="left"), A


def decision_stats(keep, A, cty1, mask1):
    """per country: predicted matches per S1, empty share, mean calibrated p of accepted pairs, records assigned."""
    out = {}
    a = keep["a"].to_numpy()
    npred = np.bincount(a, minlength=len(cty1))
    for c in sorted(set(cty1[mask1].tolist())):
        m = mask1 & (cty1 == c)
        km = m[a]
        out[c] = {"s1": int(m.sum()), "pred_per_s1": float(npred[m].mean()), "empty_share": float((npred[m] == 0).mean()),
                  "mean_p_accepted": float(keep["p"].to_numpy()[km].mean()),
                  "exp_fp_per_s1": float((1 - keep["p"].to_numpy()[km]).sum() / m.sum()),
                  "assigned_band_share": float(((A["p"].to_numpy() >= 0.05) & (A["p"].to_numpy() <= 0.95))[m[A["a"].to_numpy()]].mean())}
    return out


def main():
    gpu_init()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    s1tr, rrtr = ids("train")
    s1te, rrte = ids("test")
    c1tr, c1te = s1tr["country"].to_numpy(), s1te["country"].to_numpy()
    crtr, crte = record_country("train"), record_country("test")
    drop = np.load(os.path.join(CACHE, "drop_19.npy"))
    sc = Scorer("train", drop)
    val = sc.fold == 0

    with Timer("1. prior per country"):
        full_true = np.bincount(Scorer("train").T["a"].to_numpy(), minlength=len(c1tr))   # no simulated drop
        n_true = sc.n_true
        pr = {}
        for c in sorted(set(c1te.tolist())):
            d = {"test_s1": int((c1te == c).sum()), "test_records": int((crte == c).sum())}
            d["test_records_per_s1"] = d["test_records"] / max(d["test_s1"], 1)
            if (c1tr == c).any():
                m = c1tr == c
                d.update({"train_s1": int(m.sum()), "train_records": int((crtr == c).sum()),
                          "train_true_per_s1": float(full_true[m].mean()),
                          "train_records_per_s1": float((crtr == c).sum() / m.sum()),
                          "train_distractor_share": 1 - float(full_true[m].sum() / (crtr == c).sum()),
                          "val_sim_true_per_s1": float(n_true[val & m].mean()),
                          "val_sim_singleton_share": float((n_true[val & m] == 0).mean())})
                # distractor share implied for test if true matches per S1 stay as in train
                d["test_implied_distractor_share"] = 1 - d["train_true_per_s1"] / d["test_records_per_s1"]
            pr[c] = d
        rep["prior"] = pr
        print(json.dumps(pr, indent=1), flush=True)

    with Timer("2. decisions val vs test (V2 = stage 2, V3 = GNN-MLX)"):
        systems = {"V2": ("oof_xgb_c2_blend_s2_v2.parquet", "feats/test_scores_final_v2.parquet", "p", False),
                   "V3": ("oof_gnn_mlx_v2.parquet", "gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True)}
        rep["decisions"] = {}
        for name, (oofn, testn, col, is_logit) in systems.items():
            iso = iso_from(oofn)
            oof = pl.read_parquet(os.path.join(CACHE, "feats", oofn)).select(["a", "b", "p"])
            keep, A = decode(oof, iso)
            m, f = sc.metrics(keep.select(["a", "b"]), val)
            dv = decision_stats(keep, A, c1tr, val)
            for c in dv:
                mc = val & (c1tr == c)
                dv[c]["true_per_s1"] = float(sc.n_true[mc].mean())
                mm, _ = sc.metrics(keep.select(["a", "b"]), mc)
                dv[c].update({"f05": mm["macro_f05"], "pair_p": mm["pair_precision"], "pair_r": mm["pair_recall"]})
            T = pl.read_parquet(os.path.join(CACHE, testn)).select(["a", "b", col]).rename({col: "p"})
            if is_logit:
                T = T.with_columns((1 / (1 + (-pl.col("p")).exp())).cast(pl.Float32).alias("p"))
            keep_t, At = decode(T, iso)
            dt = decision_stats(keep_t, At, c1te, np.ones(len(c1te), bool))
            rep["decisions"][name] = {"val_all": m, "val": dv, "test": dt}
            print(name, json.dumps(rep["decisions"][name], indent=1), flush=True)
            del oof, T, keep, keep_t, A, At

    with Timer("3. stage-1 shift: refit vs half models on test"):
        bl = json.load(open(os.path.join(RESULTS, "xgboost", "xgb_c2_blend_v2", "report.json")))["blend"]
        seen = set(c1tr.tolist())
        w = np.where(np.isin(c1te, list(seen)), bl["w_seen"], bl["w_unseen"]).astype(np.float32)
        files = sorted(glob.glob(os.path.join(CACHE, "feats", "c2_test", "part_*.parquet")))
        models = {}
        for tag in (bl["full"], bl["noemb"]):
            rp = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
            models[tag] = (rp["features"], {k: xgb.Booster(model_file=os.path.join(RESULTS, "xgboost", tag, f"model_{k}.json"))
                                            for k in ("A", "B")})
        for k in ("A", "B"):
            outs = []
            for f in files:
                d = pl.read_parquet(f).rename({"r_e2f_emb": "r_e3_emb"}, strict=False)
                ps = []
                for tag in (bl["full"], bl["noemb"]):
                    feats, mods = models[tag]
                    X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
                    ps.append(mods[k].predict(xgb.DMatrix(X)))
                ww = w[d["a"].to_numpy()]
                outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", (ww * ps[0] + (1 - ww) * ps[1]).astype(np.float32))))
            H = pl.concat(outs)
            H.write_parquet(os.path.join(CACHE, "feats", f"test_stage1_half{k}_v2.parquet"))
            del H, outs
        R = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_final_v2.parquet"))
        HA = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_halfA_v2.parquet"))
        HB = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_halfB_v2.parquet"))
        assert (R["b"].to_numpy() == HA["b"].to_numpy()).all() and (R["a"].to_numpy() == HA["a"].to_numpy()).all()
        pR, pA, pB = R["p"].to_numpy(), HA["p"].to_numpy(), HB["p"].to_numpy()
        ca = c1te[R["a"].to_numpy()]
        oof = pl.read_parquet(os.path.join(CACHE, "feats", "oof_xgb_c2_blend_v2.parquet"), columns=["a", "fold", "p"])
        po, fo, ao = oof["p"].to_numpy(), oof["fold"].to_numpy(), oof["a"].to_numpy()
        sh = {}
        for c in sorted(set(c1te.tolist())):
            m = ca == c
            d = {}
            for nm, p in (("refit", pR), ("halfA", pA), ("halfB", pB)):
                q = p[m]
                d[nm] = {"band_share": float(((q >= BAND[0]) & (q <= BAND[1])).mean()), "share_gt_0.5": float((q > 0.5).mean()),
                         "mean_abs_logit_hi": float(np.abs(logit(q[q > 0.5])).mean())}
            d["mean_abs_logit_diff_refit_vs_halfA"] = float(np.abs(logit(pR[m]) - logit(pA[m])).mean())
            d["mean_abs_logit_diff_halfB_vs_halfA"] = float(np.abs(logit(pB[m]) - logit(pA[m])).mean())
            mv = (fo == 0) & (c1tr[ao] == c) if c in seen else None
            if mv is not None:
                q = po[mv]
                d["val_oof"] = {"band_share": float(((q >= BAND[0]) & (q <= BAND[1])).mean()), "share_gt_0.5": float((q > 0.5).mean()),
                                "mean_abs_logit_hi": float(np.abs(logit(q[q > 0.5])).mean())}
            sh[c] = d
        rep["stage1_shift"] = sh
        print(json.dumps(sh, indent=1), flush=True)
    rep["runtime_s"] = time.time() - t0
    json.dump(rep, open(os.path.join(OUT, "gap_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
