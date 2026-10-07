"""Write the submission from a test score table (any re-scorer: GNN, fused stage 2 ...).

The decision rule and its settings come from the validation run (decode_eval.json of that scorer): the rule with the
better tuning-fold score is used; isotonic calibration is fitted on the scorer's out-of-fold table (folds 1-2).
candidate_pairs.tsv = every scored test pair (the full candidate table), grouped by S1.

  python code/final/final_from_scores.py --test_scores gnn/gnn_v2_test_scores.parquet --col gnn --logit \
      --oof oof_gnn_v2.parquet --decode results/gnn/gnn_v2/decode_eval.json --out final_v3
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import best_per_record, decide_records, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_scores", required=True, help="path under data/cache with a, b, <col>")
    ap.add_argument("--col", required=True)
    ap.add_argument("--logit", action="store_true")
    ap.add_argument("--oof", required=True, help="oof table under data/cache/feats (a, b, fold, y, p)")
    ap.add_argument("--decode", required=True, help="decode_eval.json of this scorer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    odir = os.path.join(RESULTS, "final", args.out, "output")
    os.makedirs(odir, exist_ok=True)
    T = pl.read_parquet(os.path.join(CACHE, args.test_scores)).select(["a", "b", args.col])
    p = T[args.col].to_numpy().astype(np.float64)
    if args.logit:
        p = 1 / (1 + np.exp(-p))
    T = T.select(["a", "b"]).with_columns(pl.Series("p", p.astype(np.float32)))
    dj = json.load(open(args.decode))
    use_thr = dj["threshold_rule"]["f05_tune"] > dj["expected_f"]["best"]["f05_tune"]
    with Timer("decide"):
        if use_thr:
            r = dj["threshold_rule"]
            keep = decide_records(best_per_record(T), r["thr"], r["margin"])
            rule = {"rule": "threshold+margin", "thr": r["thr"], "margin": r["margin"], "f05_tune": r["f05_tune"]}
        else:
            oof = pl.read_parquet(os.path.join(CACHE, "feats", args.oof))
            tr = oof.filter(pl.col("fold").is_in([1, 2]))
            tr = tr.sample(n=min(8_000_000, tr.height), seed=0)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(tr["p"].to_numpy(), tr["y"].to_numpy())
            T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
            b = dj["expected_f"]["best"]
            keep, _ = expected_f_decode(assign(T, floor=0.01), gamma=b["gamma"], extra=b["extra"], M=4096)
            rule = {"rule": "expected-F0.5", **b}
    with Timer("write files"):
        s1, rr = ids("test")
        s1id, rid = s1["id"].to_numpy(), rr["id"].to_numpy()
        cand = {s1id[a]: [rid[x] for x in bs] for a, bs in T.group_by("a").agg(pl.col("b")).iter_rows()}
        match = keep.group_by("a").agg(pl.col("b"))
        ml = {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()}
        order = s1id.tolist()
        write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order, cand, "candidate_entity_ids")
        write_id_lists(os.path.join(odir, "matching_results.tsv"), order, ml, "matched_entity_ids")
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    stats = {"rule": rule, "test_s1": len(order), "candidate_pairs": T.height, "pred_matches": keep.height,
             "s1_empty": int((~has).sum()), "by_country": {}, "runtime_s": time.time() - t0}
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()),
                                  "matches": int((cty[keep["a"].to_numpy()] == c).sum())}
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(os.path.join(RESULTS, "final", args.out, "test_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
