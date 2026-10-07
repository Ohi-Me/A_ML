"""Write a V5 submission from a test score table with per-country decoding.

  seen countries (US, India): isotonic (OOF folds 1-2) + expected-F0.5 with the gamma chosen on folds 1-2
  unseen countries (France) : same calibration, expected-F0.5 with gamma_unseen = the gamma that maximised F0.5 on the
                              unseen country in the leave-one-country-out runs (India -> US), read from
                              results/phase2/loco_India2US_<name>.json ('best_gamma_unseen' of the chosen system);
                              no test labels are used anywhere.

  python code/v5/final_v5.py --scores feats/test_scores_final_v5.parquet --out final_v5
  python code/v5/final_v5.py --scores feats/test_scores_final_v5pl.parquet --out final_v5pl
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
from common.decode import assign, expected_f_decode  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402
from phase2.p2lib import device, iso_from_oof  # noqa: E402


def loco_gamma(name, system):
    for d in (f"loco_India2US_{name}", f"loco_India2US_noemb_{name}"):
        p = os.path.join(RESULTS, "phase2", f"{d}.json")
        if os.path.exists(p):
            s = json.load(open(p))["systems"]
            for sysname in (system, "stage1_all", "stage1"):
                if sysname in s and "best_gamma_unseen" in s[sysname]:
                    return float(s[sysname]["best_gamma_unseen"]), f"{d}:{sysname}"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="feats/test_scores_final_v5.parquet")
    ap.add_argument("--oof", default="oof_xgb_v5_s2.parquet")
    ap.add_argument("--decode", default=os.path.join(RESULTS, "xgboost", "xgb_v5_s2", "decode_eval.json"))
    ap.add_argument("--gamma_unseen", default="loco", help="a number, or 'loco' (from the LOCO run), or 'same'")
    ap.add_argument("--loco_name", default="v5")
    ap.add_argument("--loco_system", default="stage1_pl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = device()
    t0 = time.time()
    dj = json.load(open(args.decode))
    b = dj["expected_f"]["best"]
    g_seen, extra = float(b["gamma"]), float(b["extra"])
    src = "decode_eval"
    if args.gamma_unseen == "same":
        g_un = g_seen
    elif args.gamma_unseen == "loco":
        g_un, src = loco_gamma(args.loco_name, args.loco_system)
        if g_un is None:
            g_un, src = g_seen, "no LOCO result: same as seen"
    else:
        g_un, src = float(args.gamma_unseen), "command line"
    T = pl.read_parquet(os.path.join(CACHE, args.scores)).select(["a", "b", "p"])
    iso = iso_from_oof(pl.read_parquet(os.path.join(CACHE, "feats", args.oof)))
    Tc = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    A = assign(Tc, floor=0.01)
    s1, rr = ids("test")
    cty = s1["country"].to_numpy()
    seen = set(ids("train")[0]["country"].unique().to_list())
    um = ~np.isin(cty, list(seen))
    ua = um[A["a"].to_numpy()]
    keeps = []
    for mask, g in ((~ua, g_seen), (ua, g_un)):
        sub = A.filter(pl.Series(mask))
        if sub.height:
            k, _ = expected_f_decode(sub, gamma=g, extra=extra, M=4096, dev=str(dev))
            keeps.append(k)
    keep = pl.concat(keeps)
    odir = os.path.join(RESULTS, "final", args.out, "output")
    os.makedirs(odir, exist_ok=True)
    s1id, rid = s1["id"].to_numpy(), rr["id"].to_numpy()
    order = s1id.tolist()
    match = keep.group_by("a").agg(pl.col("b"))
    write_id_lists(os.path.join(odir, "matching_results.tsv"), order, {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()},
                   "matched_entity_ids")
    write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order,
                   {s1id[a]: [rid[x] for x in bs] for a, bs in T.group_by("a").agg(pl.col("b")).iter_rows()}, "candidate_entity_ids")
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    stats = {"scores": args.scores, "gamma_seen": g_seen, "gamma_unseen": g_un, "gamma_unseen_source": src,
             "pred_matches": keep.height, "s1_empty": int((~has).sum()), "by_country": {}, "runtime_s": time.time() - t0}
    ka = keep["a"].to_numpy()
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()), "matches": int(m[ka].sum()),
                                  "matches_per_s1": float(m[ka].sum() / m.sum())}
    json.dump(stats, open(os.path.join(RESULTS, "final", args.out, "test_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
