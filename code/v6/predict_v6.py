"""V6 test scoring: replays on test exactly the chain that was trained on the density-matched universe.

  stage 1   full + no-embedding XGBoost refit on all folds of the universe (x1.25 rounds), blended 0.75 / 0.25
            (0.5 / 0.5 for countries without training labels, as in V2)
  round 1   context + sibling + collective evidence (from stage-1 p) + XLM-R score -> round-1 model refit on all folds
  round 2   collective evidence recomputed from the round-1 p (+ prev_p) -> round-2 model refit on all folds
The extra score columns (XLM-R, optional second cross-encoder) are the ones the round models were trained with; their
test tables are the train table names with _train_scores -> _test_scores.
Writes feats/test_stage1_<out>.parquet and feats/test_scores_<out>.parquet (a, b, p). The refit models are cached in
results/final/<out>/refit_<tag>.json, so a re-run (or code/v6/pl_v6.py) reuses them. Decoding and the submission
files: code/v6/final_v6.py.

  python code/v6/predict_v6.py --step stage1 --out final_v6     (then the second cross-encoder scores the test band)
  python code/v6/predict_v6.py --step rounds --out final_v6
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import ids  # noqa: E402
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.xgbdata import PartIter, pair_files  # noqa: E402
from phase2.p2lib import device  # noqa: E402
from v6.collective import COLS, collective_features, record_attrs  # noqa: E402
from v6.train_collective import build_side  # noqa: E402

REN = {"r_e2f_emb": "r_e3_emb"}          # the test tables name the retrieval-rank column after the e2f encoder


def refit(files, tag, rdir, mult, folds=(0, 1, 2, 3, 4)):
    """refit a run on all folds of the universe with its own features / params and rounds x mult (cached)."""
    rp = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
    path = os.path.join(rdir, f"refit_{tag}.json")
    if os.path.exists(path):
        print("reuse", path, flush=True)
        return xgb.Booster(model_file=path), rp["features"]
    n = max(int(rp["rounds"] * mult), 1)
    with Timer(f"refit {tag} on all folds ({n} rounds)"):
        d = xgb.QuantileDMatrix(PartIter(files, rp["features"], list(folds)), max_bin=int(rp["params"].get("max_bin", 256)))
        m = xgb.train(rp["params"], d, n)
        del d
        m.save_model(path)
    return m, rp["features"]


def read_test_part(f):
    d = pl.read_parquet(f)
    return d.rename({k: v for k, v in REN.items() if k in d.columns})


def score_parts(files, models, side=None, rows=None):
    """stream the test parts (+ the aligned rows of a side frame) through one or more models.
    models: list of (booster, features). rows: optional bool mask over all test pairs (score only those).
    Returns (a, b) of the scored rows and one probability array per model."""
    outs, ab, off = [[] for _ in models], [], 0
    for f in files:
        d = read_test_part(f)
        n = d.height
        if side is not None:
            add = side[off:off + n]
            assert (add["a"].to_numpy() == d["a"].to_numpy()).all() and (add["b"].to_numpy() == d["b"].to_numpy()).all()
            keep = [c for c in add.columns if c not in ("r", "a", "b", "fold", "y")]
            d = pl.concat([d, add.select(keep).rename({"p": "s1_p"})], how="horizontal")
        if rows is not None:
            d = d.filter(pl.Series(rows[off:off + n]))
        off += n
        ab.append(d.select(["a", "b"]))
        for k, (m, feats) in enumerate(models):
            outs[k].append(m.predict(xgb.DMatrix(d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy())))
    return pl.concat(ab), [np.concatenate(o).astype(np.float32) for o in outs]


def stage1_blend(AB, p_full, p_noemb, bl, cty, seen):
    w = np.where(np.isin(cty[AB["a"].to_numpy()], list(seen)), bl["w_seen"], bl["w_unseen"]).astype(np.float32)
    return AB.with_columns(pl.Series("p", (w * p_full + (1 - w) * p_noemb).astype(np.float32)))


def collective_rounds(D1, trf, tef, args, rdir, dev, attrs, extras):
    """rounds 1 and 2 on test from the stage-1 table D1 (a, b, p in test-part order). Returns (p_round1, p_round2)."""
    D1 = D1.select(["a", "b", "p"]).with_row_index("r")
    with Timer("round-1 features on test"):
        S1side = build_side(D1, None, extras, "test", args.etag, args.btag, dev, attrs)
    m1 = refit(pair_files(trf, os.path.join(CACHE, "feats", f"{args.feats}_{args.c1}")), args.c1, rdir, args.round_mult)
    with Timer("round-1 model on test"):
        _, (p_r1,) = score_parts(tef, [m1], side=S1side)
    with Timer("round-2 features on test"):
        S2side = S1side.drop(COLS).with_columns(pl.Series("prev_p", p_r1))
        del S1side
        C = collective_features(S2side.select(["r", "a", "b", pl.col("prev_p").alias("p")]), attrs)
        S2side = S2side.join(C.select(["r"] + COLS), on="r", how="left").sort("r")
        del C
    m2 = refit(pair_files(trf, os.path.join(CACHE, "feats", f"{args.feats}_{args.c2}")), args.c2, rdir, args.round_mult)
    with Timer("round-2 model on test"):
        _, (p_r2,) = score_parts(tef, [m2], side=S2side)
    return p_r1, p_r2


def test_extras(c1, c2):
    """the extra score columns the collective models were trained with, as test tables (train path -> test path)."""
    ex = [json.load(open(os.path.join(RESULTS, "xgboost", t, "report.json")))["args"]["extra"] for t in (c1, c2)]
    assert ex[0] == ex[1], f"rounds 1 and 2 were trained with different extra inputs: {ex}"
    out = []
    for x in [x for x in ex[0].split(",") if x]:
        path, col = x.split(":")
        tp = path.replace("_train_scores", "_test_scores")
        assert tp != path and os.path.exists(os.path.join(CACHE, tp)), f"test table for the extra input {x} missing: {tp}"
        out.append((tp, col))
    return out


def add_args(ap):
    ap.add_argument("--feats", default="c2v6_train_dmsA", help="training universe (feature parts under feats/)")
    ap.add_argument("--test_feats", default="c2v6_test")
    ap.add_argument("--s1", default="xgb_v6_blend", help="blend run (its report names the full / no-embedding runs)")
    ap.add_argument("--c1", default="xgb_v6_c1")
    ap.add_argument("--c2", default="xgb_v6_c2")
    ap.add_argument("--etag", default="e2f")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--round_mult", type=float, default=1.25)
    ap.add_argument("--out", default="final_v6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=["stage1", "rounds", "all"],
                    help="stage1: test stage-1 scores only (the band of a new cross-encoder is taken from them); "
                         "rounds: collective rounds from the saved stage-1 scores")
    add_args(ap)
    args = ap.parse_args()
    dev = device()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "final", args.out)
    os.makedirs(rdir, exist_ok=True)
    trf = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    tef = sorted(glob.glob(os.path.join(CACHE, "feats", args.test_feats, "part_*.parquet")))
    assert trf and tef, (args.feats, args.test_feats)
    s1path = os.path.join(CACHE, "feats", f"test_stage1_{args.out}.parquet")
    if args.step in ("stage1", "all"):
        bl = json.load(open(os.path.join(RESULTS, "xgboost", args.s1, "report.json")))["blend"]
        cty = ids("test")[0]["country"].to_numpy()
        seen = set(ids("train")[0]["country"].unique().to_list())
        models = [refit(trf, bl["full"], rdir, args.round_mult), refit(trf, bl["noemb"], rdir, args.round_mult)]
        with Timer("stage 1 on test"):
            AB, (pf, pn) = score_parts(tef, models)
            stage1_blend(AB, pf, pn, bl, cty, seen).write_parquet(s1path)
        if args.step == "stage1":
            return
    cty = ids("test")[0]["country"].to_numpy()
    D1 = pl.read_parquet(s1path)
    extras = test_extras(args.c1, args.c2)
    print("extra inputs on test:", extras, flush=True)
    p_r1, p_r2 = collective_rounds(D1, trf, tef, args, rdir, dev, record_attrs("test"), extras)
    T = D1.select(["a", "b"]).with_columns(pl.Series("p", p_r2), pl.Series("p_round1", p_r1), pl.Series("p_stage1", D1["p"]))
    T.write_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.out}.parquet"))
    ac = cty[T["a"].to_numpy()]
    rep = {"pairs": T.height, "runtime_s": time.time() - t0, "args": vars(args), "extras": extras, "mean_p_by_country": {
        c: {k: float(T[k].to_numpy()[ac == c].mean()) for k in ("p_stage1", "p_round1", "p")} for c in sorted(set(ac.tolist()))}}
    json.dump(rep, open(os.path.join(rdir, "predict_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
