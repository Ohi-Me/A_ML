"""Apply the existing V2 chain (stage-1 full + no-embedding halves -> blend -> stage-2 halves) to another labelled
universe (e.g. the density-matched simulation) and compare it with SIM19, exactly as the chain is applied to test:
calibration and decoder settings come from the SIM19 validation run. Answers: does the test-US gap reproduce on
labelled data when only the density changes? And how much does re-calibrating on the new universe recover?

  python code/v6/score_chain.py --feats c2_train_dmsA --code 62 --name dmsA
     -> feats/oof_v2chain_<name>.parquet, results/v6/score_chain_<name>.json
Also reported on the universe: the "V3 without GNN" chain (stage-1 blend + XLM-R logistic stack, the phase-2 ablation's
most consistent system) exactly as it is applied to test, per-country EM prior correction of the V2 chain, and the
ceiling (every true pair among the candidates accepted, nothing else) = the best any scorer can do on these candidates.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from phase2.p2lib import cpath, decode, device, iso_from_oof, logit, uncertain_share, val_report  # noqa: E402
from v6.prior_em import em_decode, train_prior  # noqa: E402

BLEND, S2 = "xgb_c2_blend_v2", "xgb_c2_blend_s2_v2"
XTR = "xenc/ml_hardv2_train_scores.parquet"
V3_DEC = "gnn/gnn_mlx_v2/decode_eval.json"


def stack_design(p, x):
    has = ~np.isnan(x)
    x0 = np.where(has, x, 0.0)
    return np.column_stack([logit(p), x0, has.astype(float), logit(p) * x0]), has


def v3nognn_stack():
    """the phase-2 'V3 without GNN' stack: logistic [logit p, x, has, logit p * x] on V2 stage-1 blend OOF + XLM-R
    (band pairs), fitted on SIM19 folds 1-2. Returns the model and its SIM19 OOF table (for the calibration)."""
    from sklearn.linear_model import LogisticRegression
    D = pl.read_parquet(cpath(f"feats/oof_{BLEND}.parquet")).join(pl.read_parquet(cpath(XTR)).select(["a", "b", "mlxenc"]),
                                                                 on=["a", "b"], how="left")
    Z, h = stack_design(D["p"].to_numpy(), D["mlxenc"].to_numpy().astype(np.float64))
    tr = h & np.isin(D["fold"].to_numpy(), [1, 2])
    lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], D["y"].to_numpy()[tr])
    return lr, apply_stack(lr, D)


def apply_stack(lr, D):
    if "mlxenc" not in D.columns:
        D = D.join(pl.read_parquet(cpath(XTR)).select(["a", "b", "mlxenc"]), on=["a", "b"], how="left")
    Z, h = stack_design(D["p"].to_numpy(), D["mlxenc"].to_numpy().astype(np.float64))
    p = D["p"].to_numpy().astype(np.float64)
    if h.any():
        p[h] = lr.predict_proba(Z[h])[:, 1]
    return D.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p.astype(np.float32)))


def halves(tag):
    import xgboost as xgb
    rp = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
    return rp["features"], {k: xgb.Booster(model_file=os.path.join(RESULTS, "xgboost", tag, f"model_{k}.json")) for k in ("A", "B")}


def score(models, d, feats):
    import xgboost as xgb
    fold = d["fold"].to_numpy()
    ia = np.isin(fold, [0, 3, 4])
    X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
    p = np.zeros(len(fold), np.float32)
    if ia.any():
        p[ia] = models["A"].predict(xgb.DMatrix(X[ia]))
    if (~ia).any():
        p[~ia] = models["B"].predict(xgb.DMatrix(X[~ia]))
    return p


def report(D, sc, iso, gamma, extra, dev, cty, em_pi=None):
    if em_pi is None:
        keep, A = decode(D.select(["a", "b", "p"]), rule="ef", iso=iso, gamma=gamma, extra=extra, dev=dev, M=4096)
        pis = None
    else:
        keep, A, pis = em_decode(D.select(["a", "b", "p"]), iso, cty, em_pi, gamma, extra, dev)
    r = val_report(sc, keep, cty)
    out = {"f0": r["f0"]["macro_f05"], "f4": r["f4"]["macro_f05"], "by_country_f0": {}}
    for c, v in r["f0"]["by_country"].items():
        m = (sc.fold == 0) & (cty == c)
        out["by_country_f0"][c] = {"f05": v["macro_f05"], "precision": v["pair_precision"], "recall": v["pair_recall"],
                                   "fp": v["fp"], "fn": v["fn"], "pred_per_s1": v["pred_per_s1"],
                                   "pred_over_true": v["n_pred_matches"] / max(int(sc.n_true[m].sum()), 1)}
    Af = A.filter(pl.Series(sc.fold[A["a"].to_numpy()] == 0))
    out["uncertain_share_f0"] = uncertain_share(Af, cty)
    if pis is not None:
        out["em_priors"] = pis
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="labelled feature table of the universe (parts under feats/)")
    ap.add_argument("--code", type=int, required=True, help="drop_<code>.npy of that universe")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    dev = device()
    t0 = time.time()
    import scipy.sparse as sp
    import torch
    from common.stage2_feats import context_features, sibling_features
    bl = json.load(open(os.path.join(RESULTS, "xgboost", BLEND, "report.json")))["blend"]
    fF, mF = halves(bl["full"])
    fN, mN = halves(bl["noemb"])
    f2, m2 = halves(S2)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    with Timer("stage 1 (V2 halves) on the universe"):
        outs = []
        for f in files:
            d = pl.read_parquet(f)
            p = bl["w_seen"] * score(mF, d, fF) + (1 - bl["w_seen"]) * score(mN, d, fN)
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p.astype(np.float32))))
        D1 = pl.concat(outs)
    with Timer("stage-2 context + sibling features"):
        D = context_features(D1.select(["a", "b", "p"]).with_row_index("r"))
        E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", "e2f_train", "emb.npy"))).to(dev)
        bdir = os.path.join(CACHE, "blocking", "b1_train")
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        D = sibling_features(D, E, NAME, ADDR, ids("train")[0].height, dev).sort("r")
        del E, NAME, ADDR
        new = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
    with Timer("stage 2 (V2 halves)"):
        outs, off = [], 0
        for f in files:
            d = pl.read_parquet(f)
            add = D[off:off + d.height]
            assert (add["b"].to_numpy() == d["b"].to_numpy()).all()
            off += d.height
            d = pl.concat([d, add.select(new).rename({"p": "s1_p"})], how="horizontal")
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", score(m2, d, f2))))
        O = pl.concat(outs)
        O.write_parquet(os.path.join(CACHE, "feats", f"oof_v2chain_{args.name}.parquet"))
    dj = json.load(open(os.path.join(RESULTS, "xgboost", S2, "decode_eval.json")))["expected_f"]["best"]
    cty = ids("train")[0]["country"].to_numpy()
    base = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{S2}.parquet"))
    iso_sim19 = iso_from_oof(base)
    rep = {"universe": args.feats, "code": args.code}
    with Timer("SIM19 reference"):
        rep["sim19_as_validated"] = report(base, Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy"))), iso_sim19,
                                           dj["gamma"], dj["extra"], dev, cty)
    sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{args.code}.npy")))
    with Timer("universe, SIM19 calibration (= how the chain is applied to test)"):
        rep["universe_sim19_calibration"] = report(O, sc, iso_sim19, dj["gamma"], dj["extra"], dev, cty)
    with Timer("universe, SIM19 calibration + per-country EM prior correction (label-free)"):
        rep["universe_sim19_calibration_em"] = report(O, sc, iso_sim19, dj["gamma"], dj["extra"], dev, cty,
                                                      em_pi=train_prior(base, iso_sim19))
    with Timer("universe, re-calibrated on its own folds 1-2"):
        rep["universe_own_calibration"] = report(O, sc, iso_from_oof(O), dj["gamma"], dj["extra"], dev, cty)
    if os.path.exists(cpath(XTR)) and os.path.exists(os.path.join(RESULTS, V3_DEC)):
        with Timer("V3 without GNN on the universe (SIM19 stack + calibration, as on test)"):
            lr, V = v3nognn_stack()
            d3 = json.load(open(os.path.join(RESULTS, V3_DEC)))["expected_f"]["best"]
            rep["v3nognn_universe_sim19_calibration"] = report(apply_stack(lr, D1), sc, iso_from_oof(V), d3["gamma"],
                                                               d3["extra"], dev, cty)
    with Timer("ceiling: every true candidate pair accepted (universe and SIM19)"):
        for key, T_, sc_ in (("ceiling", O, sc), ("ceiling_sim19", base, Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy"))))):
            r = val_report(sc_, T_.filter(pl.col("y") == 1).select(["a", "b"]), cty)
            rep[key] = {"f0": r["f0"]["macro_f05"], "f4": r["f4"]["macro_f05"],
                        "by_country_f0": {c: v["macro_f05"] for c, v in r["f0"]["by_country"].items()},
                        "pair_recall_f0": r["f0"]["pair_recall"]}
    rep["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "v6"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v6", f"score_chain_{args.name}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
