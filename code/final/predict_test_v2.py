"""Final two-stage prediction for the test split.

  stage 1 : XGBoost fitted on all training folds of the stage-1 feature table (round count from the OOF run x1.25)
  stage 2 : context + sibling features from the test stage-1 scores, XGBoost fitted on all folds of the stage-2 table
  decision: isotonic calibration fitted on the OOF stage-2 scores (folds 1-2), then expected-F0.5 decoding per S1
            with the settings chosen in decode_eval.py

  python code/final/predict_test_v2.py --s1_feats c2_train_sim19 --s1_tag xgb_c2_sim19 --s2_tag xgb_c2_sim19_s2 \
      --test_feats c2_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v2
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
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import best_per_record, decide_records, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402
from common.xgbdata import PartIter, pair_files  # noqa: E402
from common.stage2_feats import context_features, sibling_features  # noqa: E402


def fit_all(files, feats, params, rounds):
    dtr = xgb.QuantileDMatrix(PartIter(files, feats, [0, 1, 2, 3, 4]), max_bin=256)
    bst = xgb.train(params, dtr, rounds)
    del dtr
    return bst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_feats", required=True)
    ap.add_argument("--s1_tag", required=True)
    ap.add_argument("--s2_tag", default="", help="empty = stage 1 only")
    ap.add_argument("--test_feats", required=True)
    ap.add_argument("--rename", default="")
    ap.add_argument("--etag_test", default="e2f")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--round_mult", type=float, default=1.25)
    ap.add_argument("--extra_test", default="", help="fusion: path:column list (test split) joined before stage 2")
    ap.add_argument("--stage1_from", default="", help="reuse the test stage-1 scores saved by this earlier run (skips refitting)")
    ap.add_argument("--blend_from", default="", help="blend run (blend_oof.py): stage 1 = w * full + (1 - w) * no-embedding")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "final", args.out)
    odir = os.path.join(rdir, "output")
    os.makedirs(odir, exist_ok=True)
    last = args.s2_tag or args.s1_tag
    rep2 = json.load(open(os.path.join(RESULTS, "xgboost", args.s2_tag, "report.json"))) if args.s2_tag else None
    dpath = os.path.join(RESULTS, "xgboost", last, "decode_eval.json")
    dj = json.load(open(dpath)) if os.path.exists(dpath) else None
    dec = dj["expected_f"]["best"] if dj else {"gamma": 1.0, "extra": 0.0}
    # the rule with the better score on the tuning folds (1-2) is used on test
    use_thr = bool(dj) and dj["threshold_rule"]["f05_tune"] > dj["expected_f"]["best"]["f05_tune"]
    thr_rule = dj["threshold_rule"] if use_thr else None
    ren = dict(x.split(":") for x in args.rename.split(",") if x)
    te_files = sorted(glob.glob(os.path.join(CACHE, "feats", args.test_feats, "part_*.parquet")))
    s1_files = sorted(glob.glob(os.path.join(CACHE, "feats", args.s1_feats, "part_*.parquet")))
    s2_files = pair_files(s1_files, os.path.join(CACHE, "feats", f"{args.s1_feats}_{args.s2_tag}")) if args.s2_tag else None

    def stage1_scores(tag):
        rep = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
        feats = rep["features"]
        with Timer(f"stage 1 ({tag}): fit on all folds"):
            m = fit_all(s1_files, feats, rep["params"], int(rep["rounds"] * args.round_mult))
            m.save_model(os.path.join(rdir, f"stage1_{tag}_all_folds.json"))
        with Timer(f"stage 1 ({tag}): score test"):
            outs = []
            for f in te_files:
                d = pl.read_parquet(f)
                d = d.rename({k: v for k, v in ren.items() if k in d.columns})
                X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
                outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", m.predict(xgb.DMatrix(X)))))
        return pl.concat(outs)

    if args.stage1_from:
        D = pl.read_parquet(os.path.join(CACHE, "feats", f"test_stage1_{args.stage1_from}.parquet"))
    elif args.blend_from:
        bl = json.load(open(os.path.join(RESULTS, "xgboost", args.blend_from, "report.json")))["blend"]
        P1 = stage1_scores(bl["full"])
        P2 = stage1_scores(bl["noemb"]).rename({"p": "p2"})
        # countries with training labels get w_seen, any other country label (open set) gets w_unseen
        seen = set(ids("train")[0]["country"].unique().to_list())
        cty = ids("test")[0]["country"].to_numpy()
        w = np.where(np.isin(cty, list(seen)), bl["w_seen"], bl["w_unseen"]).astype(np.float32)
        D = P1.join(P2, on=["a", "b"], how="left")
        D = D.with_columns(pl.Series("w", w[D["a"].to_numpy()])).with_columns(
            (pl.col("w") * pl.col("p") + (1 - pl.col("w")) * pl.col("p2")).alias("p")).select(["a", "b", "p"])
        print("stage-1 blend weights by country:", {c: float(bl["w_seen"] if c in seen else bl["w_unseen"])
                                                     for c in sorted(set(cty.tolist()))}, flush=True)
    else:
        D = stage1_scores(args.s1_tag)
    D.write_parquet(os.path.join(CACHE, "feats", f"test_stage1_{args.out}.parquet"))   # for the uncertain-pair models
    if not args.s2_tag:
        T = D
    else:
        with Timer("stage 2: test features"):
            D = D.with_row_index("r")
            D = context_features(D)
            for ex in [x for x in args.extra_test.split(",") if x]:
                path, col = ex.split(":")
                D = D.join(pl.read_parquet(os.path.join(CACHE, path)).select(["a", "b", col]), on=["a", "b"], how="left")
                print("fusion feature", col, "coverage", float(D[col].is_not_null().mean()), flush=True)
            E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", f"{args.etag_test}_test", "emb.npy"))).to(dev)
            bdir = os.path.join(CACHE, "blocking", f"{args.btag}_test")
            NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
            ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
            n1 = ids("test")[0].height
            D = sibling_features(D, E, NAME, ADDR, n1, dev)
            del E, NAME, ADDR
            torch.cuda.empty_cache()
            D = D.sort("r")
            new_cols = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
        with Timer("stage 2: fit on all folds"):
            f2 = rep2["features"]
            m2 = fit_all(s2_files, f2, rep2["params"], int(rep2["rounds"] * args.round_mult))
            m2.save_model(os.path.join(rdir, "stage2_all_folds.json"))
        with Timer("stage 2: score test"):
            off = 0
            outs = []
            for f in te_files:
                d = pl.read_parquet(f)
                d = d.rename({k: v for k, v in ren.items() if k in d.columns})
                add = D[off:off + d.height].select(new_cols).rename({"p": "s1_p"})
                assert (D[off:off + d.height]["b"].to_numpy() == d["b"].to_numpy()).all()
                off += d.height
                d = pl.concat([d, add], how="horizontal")
                X = d.select([pl.col(c).cast(pl.Float32) for c in f2]).to_numpy()
                outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", m2.predict(xgb.DMatrix(X)))))
            T = pl.concat(outs)
            del D
    T.write_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.out}.parquet"))
    with Timer("calibrate + decode"):
        oof = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{last}.parquet"))
        tr = oof.filter(pl.col("fold").is_in([1, 2]))
        tr = tr.sample(n=min(8_000_000, tr.height), seed=0)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(tr["p"].to_numpy(), tr["y"].to_numpy())
        del oof, tr
        if use_thr:
            keep = decide_records(best_per_record(T), thr_rule["thr"], thr_rule["margin"])
            dec = {"rule": "threshold+margin", "thr": thr_rule["thr"], "margin": thr_rule["margin"]}
        else:
            T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
            keep, ks = expected_f_decode(assign(T, floor=0.01), gamma=dec["gamma"], extra=dec["extra"], M=4096)
            dec = {"rule": "expected-F0.5", **dec}
    with Timer("write submission files"):
        s1, rr = ids("test")
        s1id = s1["id"].to_numpy()
        rid = rr["id"].to_numpy()
        cand = T.group_by("a").agg(pl.col("b"))
        cand_lists = {s1id[a]: [rid[x] for x in bs] for a, bs in cand.iter_rows()}
        match = keep.group_by("a").agg(pl.col("b"))
        match_lists = {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()}
        order = s1id.tolist()
        write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order, cand_lists, "candidate_entity_ids")
        write_id_lists(os.path.join(odir, "matching_results.tsv"), order, match_lists, "matched_entity_ids")
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    stats = {"test_s1": len(order), "candidate_pairs": T.height, "pred_matches": keep.height,
             "s1_with_matches": int(has.sum()), "s1_empty": int((~has).sum()), "decode": dec, "by_country": {},
             "args": vars(args), "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()),
                                  "matches": int(keep.filter(pl.Series(cty[keep["a"].to_numpy()] == c)).height)}
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(os.path.join(rdir, "test_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
