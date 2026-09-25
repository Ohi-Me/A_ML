"""Fusion layer: GNN probability + multilingual cross-encoder score (uncertain pairs only) -> one probability.

Logistic model on [logit(p_gnn), xenc, has_xenc, logit(p_gnn) * xenc], fitted on folds 1-2 of the out-of-fold
tables (pairs that have a cross-encoder score); pairs without one keep the GNN probability.

  python code/final/fuse.py --mode train --gnn_oof oof_gnn_v2.parquet --xenc xenc/ml_hardv2_train_scores.parquet --out fuse_v2
  python code/final/fuse.py --mode test  --gnn_test gnn/gnn_v2_test_scores.parquet --xenc xenc/ml_hardv2_test_scores.parquet --out fuse_v2
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402


def lp(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(p, x):
    has = ~np.isnan(x)
    x0 = np.where(has, x, 0.0)
    return np.column_stack([lp(p), x0, has.astype(float), lp(p) * x0]), has


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "test"], required=True)
    ap.add_argument("--gnn_oof", default="oof_gnn_v2.parquet")
    ap.add_argument("--gnn_test", default="gnn/gnn_v2_test_scores.parquet")
    ap.add_argument("--xenc", required=True)
    ap.add_argument("--out", default="fuse_v2")
    args = ap.parse_args()
    gpu_init()
    rdir = os.path.join(RESULTS, "gnn", args.out)
    os.makedirs(rdir, exist_ok=True)
    X = pl.read_parquet(os.path.join(CACHE, args.xenc)).select(["a", "b", "mlxenc"])
    if args.mode == "train":
        D = pl.read_parquet(os.path.join(CACHE, "feats", args.gnn_oof)).join(X, on=["a", "b"], how="left")
        p = D["p"].to_numpy().astype(np.float64)
        Z, h = design(p, D["mlxenc"].to_numpy().astype(np.float64))
        tr = h & np.isin(D["fold"].to_numpy(), [1, 2])
        lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], D["y"].to_numpy()[tr])
        json.dump({"coef": lr.coef_.ravel().tolist(), "intercept": float(lr.intercept_[0])},
                  open(os.path.join(rdir, "fusion_logreg.json"), "w"), indent=1)
        p2 = p.copy()
        p2[h] = lr.predict_proba(Z[h])[:, 1]
        D.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p2.astype(np.float32))).write_parquet(
            os.path.join(CACHE, "feats", f"oof_{args.out}.parquet"))
        print("fitted on", int(tr.sum()), "pairs; coef", lr.coef_.ravel().tolist(), flush=True)
    else:
        w = json.load(open(os.path.join(rdir, "fusion_logreg.json")))
        T = pl.read_parquet(os.path.join(CACHE, args.gnn_test)).join(X, on=["a", "b"], how="left")
        p = 1 / (1 + np.exp(-T["gnn"].to_numpy().astype(np.float64)))
        Z, h = design(p, T["mlxenc"].to_numpy().astype(np.float64))
        z = Z @ np.array(w["coef"]) + w["intercept"]
        p2 = p.copy()
        p2[h] = 1 / (1 + np.exp(-z[h]))
        os.makedirs(os.path.join(CACHE, "fuse"), exist_ok=True)
        T.select(["a", "b"]).with_columns(pl.Series("p", p2.astype(np.float32))).write_parquet(
            os.path.join(CACHE, "fuse", f"{args.out}_test_scores.parquet"))
        print("test pairs", T.height, "with cross-encoder score", int(h.sum()), flush=True)


if __name__ == "__main__":
    main()
