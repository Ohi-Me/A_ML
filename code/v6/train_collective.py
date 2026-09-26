"""V6 stage 2/3: context + sibling + COLLECTIVE features (+ XLM-R score) on top of the stage-1 OOF probability.

Round 1: collective evidence computed from the stage-1 OOF probability.
Round 2 (--prev <round-1 tag>): collective evidence recomputed from the round-1 OOF probability, which is also
         added as a feature (prev_p). This is iterative collective classification (Lu & Getoor 2003).
Same cross-fitting as every other stage (A: folds 1,2 -> scores 0,3,4, early stop on fold 3; B: folds 3,4 -> 1,2);
all inputs are out-of-fold, so nothing leaks. Writes side-car parts next to the universe's feature parts, the OOF
table, models and report (like xgboost/stage2.py, whose data handling this follows).

  python code/v6/train_collective.py --feats c2v6_train_dmsA --s1tag xgb_v6_blend --tag xgb_v6_c1 --simdrop 62
  python code/v6/train_collective.py --feats c2v6_train_dmsA --s1tag xgb_v6_blend --prev xgb_v6_c1 --tag xgb_v6_c2 --simdrop 62
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import score_split  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, NORM, RESULTS  # noqa: E402
from common.stage2_feats import context_features, sibling_features  # noqa: E402
from common.xgbdata import PartIter, read_part  # noqa: E402
from v6.collective import COLS, collective_features, record_attrs  # noqa: E402

NON_FEATURES = {"a", "b", "fold", "y", "b_has_match"}
AUTO_EXTRAS = {"v6": [("xenc/ml_hardv2_train_scores.parquet", "mlxenc"), ("xenc/ml_v6band_train_scores.parquet", "bgexenc")],
               "v7": [("xenc/ml_v7xlmr_train_scores.parquet", "mlxenc"), ("xenc/ml_v7bge_train_scores.parquet", "bgexenc"),
                      ("xenc/ml_v7qwen_train_scores.parquet", "qwxenc")]}


def build_side(D, prev, extras, split, etag, btag, dev, attrs):
    """D: a, b, p (stage-1) [+ fold, y] with row id r. prev: frame a, b, p of the previous round (or None).
    Returns D sorted by r with context, sibling, extra, collective (+ prev_p) columns."""
    D = context_features(D)
    for path, col in extras:
        D = D.join(pl.read_parquet(os.path.join(CACHE, path)).select(["a", "b", col]), on=["a", "b"], how="left")
        print("extra feature", col, "coverage", float(D[col].is_not_null().mean()), flush=True)
    E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", f"{etag}_{split}", "emb.npy"))).to(dev)
    bdir = os.path.join(CACHE, "blocking", f"{btag}_{split}")
    NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
    ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
    n1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s1.parquet"), columns=["id"]).height
    D = sibling_features(D, E, NAME, ADDR, n1, dev)
    del E, NAME, ADDR
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    D = D.sort("r")
    if prev is not None:
        pp = D.select(["r", "a", "b"]).join(prev.select(["a", "b", pl.col("p").alias("prev_p")]), on=["a", "b"], how="left")
        assert pp["prev_p"].null_count() == 0, "previous round does not cover every pair"
        D = D.join(pp.select(["r", "prev_p"]), on="r", how="left").sort("r")
        C = collective_features(D.select(["r", "a", "b", pl.col("prev_p").alias("p")]), attrs)
    else:
        C = collective_features(D.select(["r", "a", "b", "p"]), attrs)
    D = D.join(C.select(["r"] + COLS), on="r", how="left").sort("r")
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="feature parts of the training universe (under feats/)")
    ap.add_argument("--s1tag", required=True, help="stage-1 (blend) OOF tag on the same universe")
    ap.add_argument("--prev", default="", help="OOF tag of the previous collective round (round 2)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--etag", default="e2f")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--simdrop", type=int, default=62)
    ap.add_argument("--extra", default="auto", help="path:column list of extra pair scores; 'auto' = XLM-R base (V3) + the "
                                                    "second cross-encoder of V6 (xenc/ml_v6band_train_scores.parquet) if it exists")
    ap.add_argument("--extra_set", default="v6", choices=["v6", "v7"], help="which score tables 'auto' looks for")
    ap.add_argument("--drop_feats", default="cos_e3,b_gap_cos_e3,b_rank_cos_e3,a_gap_cos_e3,a_rank_cos_e3,b_gap2_cos_e3,"
                                            "r_e3_emb,sib_emb_max,sib_emb_mean")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.05)
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "xgboost", args.tag)
    os.makedirs(rdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    out_dir = os.path.join(CACHE, "feats", f"{args.feats}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    if args.extra == "auto":
        args.extra = ",".join(f"{p}:{c}" for p, c in AUTO_EXTRAS[args.extra_set] if os.path.exists(os.path.join(CACHE, p)))
    extras = [tuple(x.split(":")) for x in args.extra.split(",") if x]
    print("extra inputs:", extras, flush=True)
    with Timer("context + sibling + collective features"):
        D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.s1tag}.parquet")).with_row_index("r")
        prev = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.prev}.parquet")) if args.prev else None
        D = build_side(D, prev, extras, "train", args.etag, args.btag, dev, record_attrs("train"))
        new_cols = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
        print("collective feature means:", {c: round(float(D[c].mean()), 4) for c in COLS}, flush=True)
    with Timer("write side-car parts"):
        off = 0
        for i, f in enumerate(files):
            n = pl.scan_parquet(f).select(pl.len()).collect().item()
            sl = D[off:off + n]
            chk = pl.read_parquet(f, columns=["a", "b"])
            assert (chk["a"].to_numpy() == sl["a"].to_numpy()).all() and (chk["b"].to_numpy() == sl["b"].to_numpy()).all()
            sl.select(new_cols).rename({"p": "s1_p"}).write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
            off += n
        assert off == D.height, (off, D.height)
        del D
    s2files = list(zip(files, sorted(glob.glob(os.path.join(out_dir, "part_*.parquet")))))
    cols = pl.scan_parquet(files).collect_schema().names() + pl.scan_parquet(s2files[0][1]).collect_schema().names()
    drop = set(x for x in args.drop_feats.split(",") if x)
    feats = [c for c in dict.fromkeys(cols) if c not in NON_FEATURES and c not in drop]
    print(len(feats), "features", flush=True)
    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"], "tree_method": "hist",
              "device": "cuda", "max_depth": args.depth, "eta": args.eta, "subsample": 0.8, "colsample_bytree": 0.8,
              "min_child_weight": 5, "lambda": 2.0, "max_bin": 256}
    models = {}
    with Timer("model A (folds 1,2; early stop fold 3)"):
        dtr = xgb.QuantileDMatrix(PartIter(s2files, feats, [1, 2]), max_bin=256)
        des = xgb.QuantileDMatrix(PartIter(s2files, feats, [3]), ref=dtr)
        bst = xgb.train(params, dtr, args.rounds, evals=[(des, "fold3")], early_stopping_rounds=150, verbose_eval=250)
        n_rounds = bst.best_iteration + 1
        models["A"] = bst[:n_rounds]
        del dtr, des
    with Timer("model B (folds 3,4)"):
        dtr = xgb.QuantileDMatrix(PartIter(s2files, feats, [3, 4]), max_bin=256)
        models["B"] = xgb.train(params, dtr, n_rounds)
        del dtr
    for k, m in models.items():
        m.save_model(os.path.join(rdir, f"model_{k}.json"))
    imp = models["A"].get_score(importance_type="gain")
    imp = sorted(((feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v) for k, v in imp.items()), key=lambda t: -t[1])
    with Timer("OOF scores"):
        outs = []
        for f in s2files:
            d = read_part(f, columns=["a", "b", "fold", "y"] + feats)
            fold = d["fold"].to_numpy()
            X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            p = np.zeros(len(fold), np.float32)
            ia = np.isin(fold, [0, 3, 4])
            if ia.any():
                p[ia] = models["A"].predict(xgb.DMatrix(X[ia]))
            if (~ia).any():
                p[~ia] = models["B"].predict(xgb.DMatrix(X[~ia]))
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p)))
        O = pl.concat(outs)
        O.write_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))
    with Timer("decision search"):
        dm = np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None
        res = score_split(O, "train", rdir, tune_folds=(1, 2), drop=dm)
    rep = {"val": res["val"], "best": res["best"], "rounds": n_rounds, "importance_gain": imp[:80], "params": params,
           "features": feats, "args": vars(args), "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    print("VALIDATION fold 0:", json.dumps(res["val"]), flush=True)
    json.dump(rep, open(os.path.join(rdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
