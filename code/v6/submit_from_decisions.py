"""Write submission files from a cached decision table (a, b) - e.g. the 'V3 without GNN' system that the phase-2
ablation already decoded on test (data/cache_v2/p2/decisions/V3_noGNN_test.parquet): stage-1 blend + XLM-R stack,
isotonic + expected-F0.5. It had validation F0.5 0.98951 (V2 0.98696) and the most consistent US behaviour
(test/validation predicted matches per S1 +0.6 %, V2 +0.95 %, V3 +2.9 %), so it is the best untested candidate.

  python code/v6/submit_from_decisions.py --decisions p2/decisions/V3_noGNN_test.parquet --out final_v3nognn
  bash code/final/validate.sh results/final/final_v3nognn/output
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import ids  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True, help="parquet under the cache with a, b (test)")
    ap.add_argument("--cands", default="feats/test_stage1_final_v2.parquet", help="every scored test pair (a, b)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keep = pl.read_parquet(os.path.join(CACHE, args.decisions)).select(["a", "b"])
    C = pl.read_parquet(os.path.join(CACHE, args.cands), columns=["a", "b"])
    miss = keep.join(C, on=["a", "b"], how="anti").height
    assert miss == 0, f"{miss} decided pairs are not in the candidate table"
    s1, rr = ids("test")
    s1id, rid = s1["id"].to_numpy(), rr["id"].to_numpy()
    order = s1id.tolist()
    odir = os.path.join(RESULTS, "final", args.out, "output")
    os.makedirs(odir, exist_ok=True)
    match = keep.group_by("a").agg(pl.col("b"))
    write_id_lists(os.path.join(odir, "matching_results.tsv"), order, {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()},
                   "matched_entity_ids")
    write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order,
                   {s1id[a]: [rid[x] for x in bs] for a, bs in C.group_by("a").agg(pl.col("b")).iter_rows()}, "candidate_entity_ids")
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    ka = keep["a"].to_numpy()
    stats = {"decisions": args.decisions, "pred_matches": keep.height, "s1_empty": int((~has).sum()), "by_country": {}}
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "matches": int(m[ka].sum()), "matches_per_s1": float(m[ka].sum() / m.sum()),
                                  "empty_rate": float((~has[m]).mean())}
    json.dump(stats, open(os.path.join(RESULTS, "final", args.out, "test_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
