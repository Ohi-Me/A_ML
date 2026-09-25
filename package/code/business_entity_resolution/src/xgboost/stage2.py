"""Stage 2: collective re-scoring with the out-of-fold stage-1 probability in context.

New features per pair (a, b):
  record side : p, best / second p of the record, gap to the best, rank of this S1 for the record
  S1 side     : sum of p over the S1's candidates, count with p > 0.5, rank of this record for the S1,
                best p among the S1's other candidates, number of records whose best S1 is this one (claimed)
  siblings    : similarity of the record to the S1's other confident records (p > 0.5): max / mean of the
                embedding cosine and of the tf-idf name / address cosine. Records of one entity look alike.
Same cross-fitting halves as stage 1 (A: folds 1,2 -> scores 0,3,4; B: folds 3,4 -> scores 1,2); stage-1 p is
out-of-fold everywhere, so nothing leaks.

  python code/xgboost/stage2.py --feats c1_train --s1tag xgb_c1 --tag xgb_c1_s2 --etag e2
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import xgboost as xgb

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import score_split  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, NORM  # noqa: E402
from common.stage2_feats import context_features, sibling_features  # noqa: E402
from common.xgbdata import PartIter, read_part  # noqa: E402

NON_FEATURES = {"a", "b", "fold", "y", "b_has_match"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--s1tag", required=True, help="stage-1 run (oof scores)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--etag", default="e2")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--keep_stage1", default="all", help="'all' or comma list of stage-1 features to keep")
    ap.add_argument("--drop_feats", default="", help="comma list of features left out of stage 2")
    ap.add_argument("--extra", default="", help="fusion: comma list of path:column under data/cache joined on (a, b)")
    ap.add_argument("--simdrop", type=int, default=0, help="simulated S1 drop in percent (simulate_drop.py); 0 = none")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "xgboost", args.tag)
    os.makedirs(rdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    out_dir = os.path.join(CACHE, "feats", f"{args.feats}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    with Timer("context + sibling features"):
        D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.s1tag}.parquet")).with_row_index("r")
        D = context_features(D)
        for ex in [x for x in args.extra.split(",") if x]:
            path, col = ex.split(":")
            D = D.join(pl.read_parquet(os.path.join(CACHE, path)).select(["a", "b", col]), on=["a", "b"], how="left")
            print("fusion feature", col, "coverage", float(D[col].is_not_null().mean()), flush=True)
        E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", f"{args.etag}_{args.split}", "emb.npy"))).to(dev)
        bdir = os.path.join(CACHE, "blocking", f"{args.btag}_{args.split}")
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        n1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s1.parquet"), columns=["id"]).height
        D = sibling_features(D, E, NAME, ADDR, n1, dev)
        del E, NAME, ADDR
        torch.cuda.empty_cache()
        D = D.sort("r")
        new_cols = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
        new_cols = ["p" if c == "p" else c for c in new_cols]
    with Timer("write stage-2 side-car parts (new columns only)"):
        off = 0
        for i, f in enumerate(files):
            n = pl.scan_parquet(f).select(pl.len()).collect().item()
            add = D[off:off + n].select(new_cols).rename({"p": "s1_p"})
            off += n
            add.write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        assert off == D.height, (off, D.height)
        del D
    s2files = list(zip(files, sorted(glob.glob(os.path.join(out_dir, "part_*.parquet")))))
    cols = pl.scan_parquet(files).collect_schema().names() + pl.scan_parquet(s2files[0][1]).collect_schema().names()
    feats = [c for c in cols if c not in NON_FEATURES]
    if args.keep_stage1 != "all":
        keep = set(args.keep_stage1.split(","))
        feats = [c for c in feats if c in keep or c.startswith(("s2_", "sib_")) or c == "s1_p"]
    feats = [c for c in feats if c not in set(x for x in args.drop_feats.split(",") if x)]
    print(len(feats), "features", flush=True)
    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"], "tree_method": "hist",
              "device": "cuda", "max_depth": args.depth, "eta": args.eta, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "lambda": 2.0, "max_bin": 256}
    models = {}
    with Timer("model A"):
        dtr = xgb.QuantileDMatrix(PartIter(s2files, feats, [1, 2]), max_bin=256)
        des = xgb.QuantileDMatrix(PartIter(s2files, feats, [3]), ref=dtr)
        bst = xgb.train(params, dtr, args.rounds, evals=[(des, "fold3")], early_stopping_rounds=150, verbose_eval=250)
        n_rounds = bst.best_iteration + 1
        models["A"] = bst[:n_rounds]
        del dtr, des
    with Timer("model B"):
        dtr = xgb.QuantileDMatrix(PartIter(s2files, feats, [3, 4]), max_bin=256)
        models["B"] = xgb.train(params, dtr, n_rounds)
        del dtr
    for k, m in models.items():
        m.save_model(os.path.join(rdir, f"model_{k}.json"))
    imp = models["A"].get_score(importance_type="gain")
    imp = sorted(((feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v) for k, v in imp.items()),
                 key=lambda t: -t[1])
    with Timer("oof scores"):
        outs = []
        for f in s2files:
            d = read_part(f, columns=["a", "b", "fold", "y"] + feats)
            fold = d["fold"].to_numpy()
            X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            p = np.zeros(len(fold), np.float32)
            ia = np.isin(fold, [0, 3, 4])
            p[ia] = models["A"].predict(xgb.DMatrix(X[ia]))
            p[~ia] = models["B"].predict(xgb.DMatrix(X[~ia]))
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p)))
        D = pl.concat(outs)
        D.write_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))
    with Timer("decision search"):
        dm = np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None
        res = score_split(D, "train", rdir, tune_folds=(1, 2), drop=dm)
    rep = {"val": res["val"], "best": res["best"], "rounds": n_rounds, "importance_gain": imp[:80], "params": params,
           "features": feats, "args": vars(args), "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    print("VALIDATION fold 0:", json.dumps(res["val"]), flush=True)
    json.dump(rep, open(os.path.join(rdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
