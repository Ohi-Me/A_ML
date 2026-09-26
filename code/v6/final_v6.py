"""Write a V6 submission from a test score table (feats/test_scores_<name>.parquet) with the decoding options of V6.

  calibration   isotonic fitted on folds 1-2 of the OOF table of the last model (--oof)
  --em          per-country Saerens EM prior correction (pi_train from that OOF table), code/v6/prior_em.py
  --caps        at most 5 Source-2 and 6 Source-3 records per S1 (training ground-truth limits), code/v6/decode_v6.py
  gamma         expected-F0.5 gamma/extra from decode_eval.json of the last model; --gamma_unseen loco takes the
                unseen country's gamma from the leave-one-country-out run (results/phase2/loco_India2US_<name>.json,
                'best_gamma_unseen'), 'same' keeps one gamma
No test labels are used anywhere. Writes results/final/<out>/output/{matching_results,candidate_pairs}.tsv and
results/final/<out>/test_stats.json.

  python code/v6/final_v6.py --scores feats/test_scores_final_v6.parquet --tag xgb_v6_c2 --caps --out final_v6
  python code/v6/final_v6.py --scores feats/test_scores_final_v6.parquet --tag xgb_v6_c2 --caps --em --out final_v6em
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import ids  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402
from phase2.p2lib import device, iso_from_oof  # noqa: E402
from v6.decode_v6 import decode_v6, record_source  # noqa: E402
from v6.prior_em import train_prior  # noqa: E402


def loco_gamma(name):
    p = os.path.join(RESULTS, "phase2", f"loco_India2US_{name}.json")
    if not os.path.exists(p):
        return None, None
    s = json.load(open(p))["systems"]
    for sysname in ("stage1_all", "stage1"):
        if sysname in s and "best_gamma_unseen" in s[sysname]:
            return float(s[sysname]["best_gamma_unseen"]), f"loco_India2US_{name}:{sysname}"
    return None, None


def write_submission(out, keep, T, stats):
    """matching_results.tsv + candidate_pairs.tsv (every S1 row, file order) and test_stats.json."""
    s1, rr = ids("test")
    s1id, rid = s1["id"].to_numpy(), rr["id"].to_numpy()
    order = s1id.tolist()
    odir = os.path.join(RESULTS, "final", out, "output")
    os.makedirs(odir, exist_ok=True)
    match = keep.group_by("a").agg(pl.col("b"))
    write_id_lists(os.path.join(odir, "matching_results.tsv"), order,
                   {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()}, "matched_entity_ids")
    write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order,
                   {s1id[a]: [rid[x] for x in bs] for a, bs in T.group_by("a").agg(pl.col("b")).iter_rows()},
                   "candidate_entity_ids")
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    ka = keep["a"].to_numpy()
    stats = {**stats, "pred_matches": keep.height, "s1_empty": int((~has).sum()), "by_country": {}}
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()), "matches": int(m[ka].sum()),
                                  "matches_per_s1": float(m[ka].sum() / m.sum()), "empty_rate": float((~has[m]).mean())}
    json.dump(stats, open(os.path.join(RESULTS, "final", out, "test_stats.json"), "w"), indent=1, default=float)
    print(out, json.dumps(stats, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="feats/test_scores_final_v6.parquet")
    ap.add_argument("--tag", default="xgb_v6_c2", help="last model: oof_<tag>.parquet and its decode_eval.json")
    ap.add_argument("--sub", default="xgboost")
    ap.add_argument("--em", action="store_true")
    ap.add_argument("--caps", action="store_true")
    ap.add_argument("--gamma_unseen", default="same", help="'same', 'loco' or a number")
    ap.add_argument("--loco_name", default="v6")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = device()
    t0 = time.time()
    T = pl.read_parquet(os.path.join(CACHE, args.scores), columns=["a", "b", "p"])
    oof = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))
    iso = iso_from_oof(oof)
    b = json.load(open(os.path.join(RESULTS, args.sub, args.tag, "decode_eval.json")))["expected_f"]["best"]
    cty = ids("test")[0]["country"].to_numpy()
    seen = set(ids("train")[0]["country"].unique().to_list())
    unseen = sorted(set(cty.tolist()) - seen)
    g_src, gmap = "decode_eval", {}
    if args.gamma_unseen == "loco":
        g, g_src = loco_gamma(args.loco_name)
        if g is None:
            g_src = "no LOCO result: same gamma"
        else:
            gmap = {c: g for c in unseen}
    elif args.gamma_unseen != "same":
        gmap, g_src = {c: float(args.gamma_unseen) for c in unseen}, "command line"
    em_pi = train_prior(oof, iso) if args.em else None
    del oof
    keep, A, info = decode_v6(T, iso, b["gamma"], b["extra"], dev, cty, src=record_source("test") if args.caps else None,
                              em_pi=em_pi, gamma_by_country=gmap)
    write_submission(args.out, keep, T, {"scores": args.scores, "tag": args.tag, "em": args.em, "caps": args.caps,
                                         "decoder": {"gamma": b["gamma"], "extra": b["extra"], "gamma_unseen_source": g_src},
                                         "decode_info": info, "runtime_s": time.time() - t0})


if __name__ == "__main__":
    main()
