"""Choose the V7 candidate passes from the miss probe, with a rule fixed before the probe has run.

Input: results/v7/miss_probe.json (code/v7/miss_probe.py, DMS universe, training labels only).
Rule (recall gains are shares of all true pairs, measured on top of the re-cut C2 depths):
  depth   x2 if it adds >= 0.0005 recall and the re-cut lists then give <= BUDGET - 2 pairs per record;
          x4 on top of x2 with the same test
  blocks  key / addr / numstreet: the largest cap whose pass adds >= 0.0002 recall at <= 1.5 pairs per record
  siblings sib_addr (groups <= 50 records) and sib_key (<= 30) if they add >= 0.0002 recall
Every chosen pass keeps its efficiency (gain / pairs per record) so build_cands.py can drop the least efficient
passes first if the total goes over the pair budget. Without a probe the conservative default below is used.

  python code/v7/cands_plan.py [--budget 12]   -> results/v7/cands_plan.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import RESULTS  # noqa: E402

DEFAULT = {"depth_mult": 2, "passes": {"key": 5, "addr": 5, "numstreet": 5, "sib_addr": 50, "sib_key": 0},
           "efficiency": {"key": 1.0, "addr": 1.0, "numstreet": 1.0, "sib_addr": 1.0, "sib_key": 0.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=12.0, help="max candidate pairs per record (C2: ~8.2)")
    args = ap.parse_args()
    path = os.path.join(RESULTS, "v7", "miss_probe.json")
    out = {"budget_pairs_per_record": args.budget}
    if not os.path.exists(path):
        out.update(DEFAULT)
        out["source"] = "default (no miss_probe.json)"
    else:
        P = json.load(open(path))
        rc, ppr = P["recall_recut"], P["pairs_per_record_recut"]
        mult = 1
        if rc["x2_depths"] - rc["c2_depths"] >= 0.0005 and ppr["x2_depths"] <= args.budget - 2:
            mult = 2
            if rc["x4_depths"] - rc["x2_depths"] >= 0.0005 and ppr["x4_depths"] <= args.budget - 2:
                mult = 4
        passes, eff = {}, {}
        for name in ("key", "addr", "numstreet"):
            best = None
            for cap, v in sorted(P["passes"][name].items(), key=lambda t: int(t[0])):
                if v["recall_gain"] >= 0.0002 and v["pairs_per_record"] <= 1.5:
                    best = (int(cap), v)
            passes[name] = best[0] if best else 0
            eff[name] = best[1]["recall_gain"] / max(best[1]["pairs_per_record"], 1e-3) if best else 0.0
        for name, cap in (("sib_addr", 50), ("sib_key", 30)):
            g = P["passes"][name]["recall_gain"]
            passes[name] = cap if g >= 0.0002 else 0
            eff[name] = g                     # cost measured by build_cands.py
        out.update({"depth_mult": mult, "passes": passes, "efficiency": eff, "source": "miss_probe.json",
                    "probe": {"recall_c2_as_filtered": P["recall_c2_as_filtered"], "recall_recut": rc,
                              "pairs_per_record_recut": ppr}})
    os.makedirs(os.path.join(RESULTS, "v7"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "v7", "cands_plan.json"), "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
