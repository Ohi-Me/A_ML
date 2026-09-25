"""Final stage: fit the pair model on all training folds, score the test candidates, write the two submission files.

  python code/final/predict_test.py --train_feats c1_train --test_feats c1_test --tag final_v1 \
         --rounds 1200 --thr 0.5 --margin 0.0

candidate_pairs.tsv = exactly the pairs the model scores (the test candidate table), grouped by S1.
matching_results.tsv = decided matches (each record to its best S1 if score >= thr and margin is met).
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
from common.decide import best_per_record, decide_records, ids  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402
from common.xgbdata import PartIter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_feats", required=True)
    ap.add_argument("--test_feats", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--features_from", required=True, help="results/xgboost/<tag>/report.json holding the feature list")
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--rename", default="", help="test column renames old:new,... (e.g. r_e2f_emb:r_e2_emb)")
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, "final", args.tag)
    odir = os.path.join(rdir, "output")
    os.makedirs(odir, exist_ok=True)
    rep_src = json.load(open(os.path.join(RESULTS, "xgboost", args.features_from, "report.json")))
    feats, params = rep_src["features"], rep_src["params"]
    tr_files = sorted(glob.glob(os.path.join(CACHE, "feats", args.train_feats, "part_*.parquet")))
    te_files = sorted(glob.glob(os.path.join(CACHE, "feats", args.test_feats, "part_*.parquet")))

    ren = dict(x.split(":") for x in args.rename.split(",") if x)
    with Timer("fit on all training folds"):
        dtr = xgb.QuantileDMatrix(PartIter(tr_files, feats, [0, 1, 2, 3, 4]), max_bin=256)
        bst = xgb.train(params, dtr, args.rounds)
        bst.save_model(os.path.join(rdir, "model_all_folds.json"))
        del dtr
    with Timer("score test candidates"):
        outs = []
        for f in te_files:
            d = pl.read_parquet(f)
            if ren:
                d = d.rename({k: v for k, v in ren.items() if k in d.columns})
            d = d.select(["a", "b"] + feats)
            X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", bst.predict(xgb.DMatrix(X)))))
        D = pl.concat(outs)
        D.write_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.tag}.parquet"))
    with Timer("write submission files"):
        s1, rr = ids("test")
        s1id = s1["id"].to_numpy()
        rid = rr["id"].to_numpy()
        pred = decide_records(best_per_record(D), args.thr, args.margin)
        cand = D.group_by("a").agg(pl.col("b"))
        cand_lists = {s1id[a]: [rid[x] for x in bs] for a, bs in cand.iter_rows()}
        match = pred.group_by("a").agg(pl.col("b"))
        match_lists = {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()}
        order = s1id.tolist()
        write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order, cand_lists, "candidate_entity_ids")
        write_id_lists(os.path.join(odir, "matching_results.tsv"), order, match_lists, "matched_entity_ids")
    stats = {"test_s1": len(order), "candidate_pairs": D.height, "pred_matches": pred.height,
             "s1_with_matches": len(match_lists), "s1_empty": len(order) - len(match_lists),
             "by_country": {}, "args": vars(args), "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()),
                                  "matches": int(pred.filter(pl.Series(cty[pred["a"].to_numpy()] == c)).height)}
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(os.path.join(rdir, "test_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
