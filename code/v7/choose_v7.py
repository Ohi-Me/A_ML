"""V7 decisions, fixed before any V7 number exists (do not tune them on the results).

All labelled numbers: DMS universe, clean held-out folds 0 and 4 (min of the two), same S1 and ground truth for V6 and
V7 (the Scorer counts every true pair, also those blocking missed, so higher candidate recall shows up as recall).

  plan (before the V7 file is written)
    round  round 2 (xgb_v7_c2) unless round 1 (xgb_v7_c1) is higher on DMS (V6: equal within 0.00001)
    caps   per-source caps if they do not lower that round's DMS score by more than 0.0001
    em, France gamma, pseudo-labels: taken from the V6 plan (results/v6/v6_plan.json; the evidence for them - the
           V2 chain under density shift and the India->US LOCO - does not depend on the candidate set)
  pick
    V7 is submitted first if DMS(V7) >= DMS(V6) + 0.0005 and, for US and India, its test / DMS-fold-0 ratio of
    predicted matches per S1 exceeds the V2 chain's ratio by at most 0.003 (the V6 rule). Order: final_v7 > final_v6
    (if it passed its own rule) > final_v3nognn > V2 (keep).
  Also reported: the candidate ceiling of V7 on DMS (every true candidate accepted) next to V6's 0.99635.

  python code/v7/choose_v7.py plan | final_args | pick
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from phase2.p2lib import device, val_report  # noqa: E402
from v6.choose_v6 import dms_scores, jload, mn, ratios, test_rate  # noqa: E402

RD = os.path.join(RESULTS, "v7")
C1, C2, V6, CODE = "xgb_v7_c1", "xgb_v7_c2", "xgb_v6_c2", 62
EPS_CAPS, EPS_PICK, TOL = 0.0001, 0.0005, 0.003


def setup():
    return device(), ids("train")[0]["country"].to_numpy(), Scorer("train", np.load(os.path.join(CACHE, f"drop_{CODE}.npy")))


def ceiling(tag, sc, cty):
    path = os.path.join(CACHE, "feats", f"oof_{tag}.parquet")
    if not os.path.exists(path):
        return None
    r = val_report(sc, pl.read_parquet(path, columns=["a", "b", "y"]).filter(pl.col("y") == 1).select(["a", "b"]), cty)
    return {"f0": r["f0"]["macro_f05"], "f4": r["f4"]["macro_f05"], "pair_recall_f0": r["f0"]["pair_recall"]}


def cmd_plan():
    dev, cty, sc = setup()
    s = {t: {c: dms_scores(t, sc, cty, dev, c) for c in (False, True)} for t in (C1, C2)}
    assert s[C2][False] is not None, "xgb_v7_c2 OOF / decode_eval missing"
    use_c1 = s[C1][False] is not None and max(mn(s[C1][False]), mn(s[C1][True])) > max(mn(s[C2][False]), mn(s[C2][True]))
    tag = C1 if use_c1 else C2
    caps = mn(s[tag][True]) >= mn(s[tag][False]) - EPS_CAPS
    v6 = (jload(RESULTS, "v6", "v6_plan.json") or {}).get("plan", {})
    plan = {"tag": tag, "col": "p_round1" if use_c1 else "p", "caps": bool(caps), "em": bool(v6.get("em", False)),
            "gamma_unseen": v6.get("gamma_unseen", "same"), "pl": False}
    out = {"rule": __doc__, "dms": {t: {("caps" if c else "no_caps"): v for c, v in d.items()} for t, d in s.items()},
           "ceiling_v7": ceiling("xgb_v7_blend", sc, cty), "plan": plan}
    os.makedirs(RD, exist_ok=True)
    json.dump(out, open(os.path.join(RD, "v7_plan.json"), "w"), indent=1, default=float)
    print(json.dumps({"plan": plan, "ceiling_v7": out["ceiling_v7"]}, default=float), flush=True)


def final_args():
    p = jload(RD, "v7_plan.json")["plan"]
    a = ["--tag", p["tag"], "--col", p["col"]] + (["--caps"] if p["caps"] else []) + (["--em"] if p["em"] else [])
    print(" ".join(a + ["--gamma_unseen", str(p["gamma_unseen"])]))


def cmd_pick():
    dev, cty, sc = setup()
    plan = jload(RD, "v7_plan.json")
    p7 = plan["plan"]
    v6plan = (jload(RESULTS, "v6", "v6_plan.json") or {}).get("plan", {})
    d7 = dms_scores(p7["tag"], sc, cty, dev, p7["caps"])
    d6 = dms_scores(V6, sc, cty, dev, bool(v6plan.get("caps", True)))
    ch = jload(RESULTS, "v6", "score_chain_dmsA.json") or {}
    v2val, v2rate = ch.get("universe_sim19_calibration"), test_rate("final_v2")
    r_v2 = ratios(v2rate, v2val) if v2val and v2rate else None
    rate7 = test_rate("final_v7")
    r7 = ratios(rate7, d7) if rate7 and d7 else None
    cons = r7 is not None and (r_v2 is None or all(r7[c] <= r_v2[c] + TOL for c in r7))
    better = d7 is not None and d6 is not None and mn(d7) >= mn(d6) + EPS_PICK
    v6c = jload(RESULTS, "v6", "v6_choice.json") or {}
    v6_ok = bool(v6c.get("checks", {}).get("final_v6", {}).get("pass", False))
    order = (["final_v7"] if (better and cons) else []) + (["final_v6"] if v6_ok else [])
    order += (["final_v3nognn"] if test_rate("final_v3nognn") else []) + ["final_v2 (keep)"]
    out = {"rule": __doc__, "plan": p7, "ceiling": {"v6_dms": (v6c.get("ceiling_dms") or {}).get("f0"),
                                                     "v7_dms": (plan.get("ceiling_v7") or {}).get("f0")},
           "dms_min_f0_f4": {"V6": mn(d6) if d6 else None, "V7": mn(d7) if d7 else None},
           "dms": {"V6": d6, "V7": d7}, "v2_test_over_dms_ratio": r_v2, "v7_test_over_dms_ratio": r7,
           "v7_better": better, "v7_consistent": cons, "submit_order": order,
           "first_choice_file": (f"results/final/{order[0]}/output/matching_results.tsv" if order[0] != "final_v2 (keep)"
                                 else "submissions/final_v2/matching_results.tsv")}
    json.dump(out, open(os.path.join(RD, "v7_choice.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rule", "dms")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    {"plan": cmd_plan, "final_args": final_args, "pick": cmd_pick}[sys.argv[1] if len(sys.argv) > 1 else "pick"]()
