"""V8 decisions, fixed before any V8 number exists (do not tune them on the results).

Same measurement as V6 / V7: DMS universe, clean held-out folds 0 and 4 (min of the two). The Scorer counts every true
pair, also those candidate generation missed, so V8's candidate recall and its pruning loss both show up.

  plan (before the V8 file is written)
    round  round 2 (xgb_v8_c2) unless round 1 (xgb_v8_c1) is higher on DMS
    caps   per-source caps if they do not lower that round's DMS score by more than 0.0001
    em, France gamma: from the V6 plan (results/v6/v6_plan.json)
  pick
    reference = the best of V6 and V7 on DMS (whichever exists). V8 is submitted first if DMS(V8) >= reference + 0.0005
    and, for US and India, its test / DMS-fold-0 ratio of predicted matches per S1 exceeds the V2 chain's by at most
    0.003 (the V6 rule). Order: final_v8 > final_v7 / final_v6 (as their own choice files say) > final_v2 (keep).
  Also reported: V8's candidate ceiling on DMS (perfect scorer on the pruned candidates), pairs per record on train and
  test, and the pruning loss on fold 0.

  python code/v8/choose_v8.py plan | final_args | pick
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

RD = os.path.join(RESULTS, "v8")
C1, C2, CODE = "xgb_v8_c1", "xgb_v8_c2", 62
REFS = {"V6": ("xgb_v6_c2", "v6", "v6_plan.json", "final_v6"), "V7": ("xgb_v7_c2", "v7", "v7_plan.json", "final_v7")}
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
    assert s[C2][False] is not None, "xgb_v8_c2 OOF / decode_eval missing"
    use_c1 = s[C1][False] is not None and max(mn(s[C1][False]), mn(s[C1][True])) > max(mn(s[C2][False]), mn(s[C2][True]))
    tag = C1 if use_c1 else C2
    caps = mn(s[tag][True]) >= mn(s[tag][False]) - EPS_CAPS
    v6 = (jload(RESULTS, "v6", "v6_plan.json") or {}).get("plan", {})
    plan = {"tag": tag, "col": "p_round1" if use_c1 else "p", "caps": bool(caps), "em": bool(v6.get("em", False)),
            "gamma_unseen": v6.get("gamma_unseen", "same"), "pl": False}
    out = {"rule": __doc__, "dms": {t: {("caps" if c else "no_caps"): v for c, v in d.items()} for t, d in s.items()},
           "ceiling_v8": ceiling("xgb_v8_blend", sc, cty), "plan": plan,
           "candidates": {k: jload(RD, f"prune_apply_{k}.json") for k in ("c4dms_train", "c4_test")},
           "pruner": {k: v for k, v in (jload(RD, "prune_c3wdms.json") or {}).items()
                      if k in ("tau", "pairs_per_record_before", "pairs_per_record_after", "by_fold")}}
    os.makedirs(RD, exist_ok=True)
    json.dump(out, open(os.path.join(RD, "v8_plan.json"), "w"), indent=1, default=float)
    print(json.dumps({"plan": plan, "ceiling_v8": out["ceiling_v8"]}, default=float), flush=True)


def final_args():
    p = jload(RD, "v8_plan.json")["plan"]
    a = ["--tag", p["tag"], "--col", p["col"]] + (["--caps"] if p["caps"] else []) + (["--em"] if p["em"] else [])
    print(" ".join(a + ["--gamma_unseen", str(p["gamma_unseen"])]))


def cmd_pick():
    dev, cty, sc = setup()
    plan = jload(RD, "v8_plan.json")
    p8 = plan["plan"]
    d8 = dms_scores(p8["tag"], sc, cty, dev, p8["caps"])
    refs = {}
    for name, (tag, sub, pfile, fin) in REFS.items():
        rp = (jload(RESULTS, sub, pfile) or {}).get("plan", {})
        t = rp.get("tag", tag)
        d = dms_scores(t, sc, cty, dev, bool(rp.get("caps", True)))
        if d is not None:
            refs[name] = {"tag": t, "dms": d, "min_f0_f4": mn(d), "file": fin}
    best_ref = max(refs.values(), key=lambda r: r["min_f0_f4"]) if refs else None
    ch = jload(RESULTS, "v6", "score_chain_dmsA.json") or {}
    v2val, v2rate = ch.get("universe_sim19_calibration"), test_rate("final_v2")
    r_v2 = ratios(v2rate, v2val) if v2val and v2rate else None
    rate8 = test_rate("final_v8")
    r8 = ratios(rate8, d8) if rate8 and d8 else None
    cons = r8 is not None and (r_v2 is None or all(r8[c] <= r_v2[c] + TOL for c in r8))
    better = d8 is not None and (best_ref is None or mn(d8) >= best_ref["min_f0_f4"] + EPS_PICK)
    order = ["final_v8"] if (better and cons) else []
    for sub, fname in (("v7", "v7_choice.json"), ("v6", "v6_choice.json")):
        c = jload(RESULTS, sub, fname) or {}
        for f in c.get("submit_order", []):
            if f not in order and not f.startswith("final_v2"):
                order.append(f)
    order.append("final_v2 (keep)")
    out = {"rule": __doc__, "plan": p8, "dms_min_f0_f4": {"V8": mn(d8) if d8 else None,
                                                          **{k: v["min_f0_f4"] for k, v in refs.items()}},
           "ceiling": {"v8_dms": (plan.get("ceiling_v8") or {}).get("f0"),
                       "v6_dms": ((jload(RESULTS, "v6", "v6_choice.json") or {}).get("ceiling_dms") or {}).get("f0")},
           "dms": {"V8": d8, **{k: v["dms"] for k, v in refs.items()}}, "v2_test_over_dms_ratio": r_v2,
           "v8_test_over_dms_ratio": r8, "v8_better": better, "v8_consistent": cons, "submit_order": order,
           "first_choice_file": (f"results/final/{order[0]}/output/matching_results.tsv" if order[0] != "final_v2 (keep)"
                                 else "submissions/final_v2/matching_results.tsv")}
    json.dump(out, open(os.path.join(RD, "v8_choice.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rule", "dms")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    {"plan": cmd_plan, "final_args": final_args, "pick": cmd_pick}[sys.argv[1] if len(sys.argv) > 1 else "pick"]()
