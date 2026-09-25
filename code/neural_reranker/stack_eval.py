"""Neural reranker, step 3: does the cross-encoder add information to the stage-1 probability?
For the uncertain pairs: logistic stacker on [logit(p), xenc] fitted on folds 1-2, applied everywhere; other pairs
keep p. Threshold + margin tuned on folds 1-2, reported on fold 0 (test-prior simulation)."""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record, decide_records  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.split import VAL_FOLD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--oof", default="xgb_c2_sim19")
ap.add_argument("--name", default="hard")
ap.add_argument("--extra", default="", help="comma list of path:column under data/cache (e.g. xenc/ml_hardv2_train_scores.parquet:mlxenc)")
ap.add_argument("--out", default="")
ap.add_argument("--simdrop", type=int, default=19)
args = ap.parse_args()
gpu_init()
D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.oof}.parquet"))
extras = [x.split(":") for x in args.extra.split(",") if x] or [[f"xenc/{args.name}_train_scores.parquet", "xenc"]]
for path, col in extras:
    D = D.join(pl.read_parquet(os.path.join(CACHE, path)).select(["a", "b", col]), on=["a", "b"], how="left")
lp = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
p = D["p"].to_numpy()
fold, y = D["fold"].to_numpy(), D["y"].to_numpy()
cols = [lp(p)]
h = np.zeros(len(p), bool)
for _, col in extras:
    v = D[col].to_numpy().astype(np.float64)
    has = ~np.isnan(v)
    h |= has
    v = np.where(has, v, 0.0)
    cols += [v, has.astype(float), lp(p) * v]
Z = np.column_stack(cols)
tr = h & np.isin(fold, [1, 2])
lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], y[tr])
p2 = p.copy()
p2[h] = lr.predict_proba(Z[h])[:, 1]
sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None)
tune, val = np.isin(sc.fold, [1, 2]), sc.fold == VAL_FOLD
out = {"coef": lr.coef_.tolist(), "uncertain_pairs": int(h.sum())}
for nm, pp in (("stage1", p), ("stage1+" + "+".join(c for _, c in extras), p2)):
    best = best_per_record(D.select(["a", "b"]).with_columns(pl.Series("p", pp)))
    grid = []
    for thr in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        for mg in (0.2, 0.3, 0.4, 0.5, 0.6):
            m, _ = sc.metrics(decide_records(best, thr, mg), tune)
            grid.append((m["macro_f05"], thr, mg))
    f, thr, mg = max(grid)
    mv, _ = sc.metrics(decide_records(best, thr, mg), val)
    out[nm] = {"thr": thr, "margin": mg, "f05_tune": f, "f05_val": mv["macro_f05"], "fp": mv["fp"], "fn": mv["fn"]}
    print(nm, out[nm], flush=True)
od = os.path.join(RESULTS, "neural_reranker", args.out or args.name)
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "stack_eval.json"), "w"), indent=1)
