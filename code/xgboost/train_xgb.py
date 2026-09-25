"""Pair classifier with XGBoost (CUDA) + F0.5 decision search.

Cross-fitting in two halves keeps memory under the 32 GB cap and gives every pair an honest score:
  model A is trained on folds {1,2} (early stopping on fold 3) and scores folds {0,3,4};
  model B is trained on folds {3,4} (same number of rounds) and scores folds {1,2}.
Fold 0 (validation) is never used for fitting. Decision thresholds are chosen on folds 1-2 and reported on fold 0.

  python code/xgboost/train_xgb.py --feats c1_train --tag xgb_c1
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

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import score_split  # noqa: E402
from common.xgbdata import PartIter  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402

NON_FEATURES = {"a", "b", "fold", "y", "b_has_match"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--tag", default="xgb_v1")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--eta", type=float, default=0.06)
    ap.add_argument("--neg_keep", type=float, default=1.0)
    ap.add_argument("--drop", default="", help="comma list of features to drop")
    ap.add_argument("--only", default="", help="comma list: use only these features")
    ap.add_argument("--fit_country", default="", help="fit only on pairs whose S1 has this country")
    ap.add_argument("--models_from", default="", help="skip training: score with model_A/B of this run")
    ap.add_argument("--simdrop", type=int, default=0, help="simulated S1 drop in percent (simulate_drop.py); 0 = none")
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "xgboost", args.tag)
    os.makedirs(rdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    cols = pl.scan_parquet(files).collect_schema().names()
    drop = set(x for x in args.drop.split(",") if x)
    feats = [c for c in cols if c not in NON_FEATURES and c not in drop]
    if args.only:
        feats = [c for c in args.only.split(",") if c]
    print(len(feats), "features:", feats, flush=True)

    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"], "tree_method": "hist",
              "device": "cuda", "max_depth": args.depth, "eta": args.eta, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "lambda": 2.0, "max_bin": 256}
    models = {}
    a_mask = None
    if args.fit_country:
        from common.decide import ids
        a_mask = ids("train")[0]["country"].to_numpy() == args.fit_country
    if args.models_from:
        src = os.path.join(RESULTS, "xgboost", args.models_from)
        feats = json.load(open(os.path.join(src, "report.json")))["features"]
        for k in ("A", "B"):
            models[k] = xgb.Booster(model_file=os.path.join(src, f"model_{k}.json"))
        n_rounds = -1
    if not args.models_from:
        with Timer("model A: train folds 1,2 / early stop fold 3"):
            it = PartIter(files, feats, [1, 2], args.neg_keep, 0, a_mask)
            dtr = xgb.QuantileDMatrix(it, max_bin=256)
            print("train rows", it.rows, "pos", it.pos, flush=True)
            des = xgb.QuantileDMatrix(PartIter(files, feats, [3], a_mask=a_mask), ref=dtr)
            bst = xgb.train(params, dtr, args.rounds, evals=[(des, "fold3")], early_stopping_rounds=150, verbose_eval=250)
            n_rounds = bst.best_iteration + 1
            models["A"] = bst[:n_rounds]
            del dtr, des
        print("rounds", n_rounds, gpu_mem(), flush=True)
        with Timer("model B: train folds 3,4"):
            dtr = xgb.QuantileDMatrix(PartIter(files, feats, [3, 4], args.neg_keep, 1, a_mask), max_bin=256)
            models["B"] = xgb.train(params, dtr, n_rounds)
            del dtr
    for k, m in models.items():
        if not args.models_from:
            m.save_model(os.path.join(rdir, f"model_{k}.json"))
    imp = models["A"].get_score(importance_type="gain")
    imp = sorted(((feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v) for k, v in imp.items()),
                 key=lambda t: -t[1])

    with Timer("out-of-fold scores (streamed)"):
        outs = []
        for f in files:
            d = pl.read_parquet(f, columns=["a", "b", "fold", "y"] + feats)
            fold = d["fold"].to_numpy()
            X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            p = np.zeros(len(fold), np.float32)
            ia = np.isin(fold, [0, 3, 4])
            if ia.any():
                p[ia] = models["A"].predict(xgb.DMatrix(X[ia]))
            ib = ~ia
            if ib.any():
                p[ib] = models["B"].predict(xgb.DMatrix(X[ib]))
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p)))
        D = pl.concat(outs)
        D.write_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))

    with Timer("decision search"):
        dm = np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None
        res = score_split(D, "train", rdir, tune_folds=(1, 2), drop=dm)
    rep = {"val": res["val"], "best": res["best"], "grid": res["grid"], "rounds": n_rounds,
           "importance_gain": imp[:80], "params": params, "args": vars(args), "features": feats,
           "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    print("VALIDATION fold 0:", json.dumps(res["val"]), flush=True)
    json.dump(rep, open(os.path.join(rdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
