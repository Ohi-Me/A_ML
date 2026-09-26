"""V6 decisions, fixed before any V6 result exists (do not tune them on the results).

Every labelled number below is measured on the density-matched universe (DMS, code/v6/simulate_universe.py): the
training data rebuilt at test density, clean held-out folds 0 and 4 (fold 3 was the early-stopping fold). Test labels
are never used.

  plan (run before the V6 submission files are written)
    caps   per-source caps (<= 5 S2 / <= 6 S3 records per S1) are used if, on the V6 OOF table, min(f0, f4) with caps
           >= without caps - 0.0001
    em     per-country EM prior correction is used if, on the V2 chain applied to DMS exactly as it is applied to test
           (score_chain.py), EM raises min(f0, f4) by > 0.001 and lowers no country's f0 by > 0.0005
    gamma  the unseen country (France) gets the leave-one-country-out gamma (India -> US inside DMS, V6 features,
           loco_eval.py) if that gamma beats the seen-country gamma on the unseen country by > 0.001; else the same
    pl     France pseudo-labels (pl_v6.py) are used if, in that LOCO run, stage1_pl beats stage1_all on the unseen
           country by > 0.001 and the pseudo-positive precision is >= 0.99
  pick (after the files exist)
    DMS score = min(f0, f4) of each system as it would run on test:
      V2 chain and V3-without-GNN chain: trained on SIM19, applied to DMS with their SIM19 calibration (score_chain)
      V6a, V6: trained and calibrated on DMS (their OOF tables), V6 with the plan's caps
    a V6 system is submitted first if DMS(V6) >= max(DMS(V2), DMS(V3 without GNN)) + 0.0005 and it is consistent:
    for US and India, (test predicted matches per S1) / (DMS fold-0 predicted per S1) may exceed the V2 chain's ratio
    by at most 0.003. Order: final_v6pl (plan.pl) > final_v6 > final_v6a (same test) > final_v3nognn > V2 (keep).

  python code/v6/choose_v6.py plan   -> results/v6/v6_plan.json
  python code/v6/choose_v6.py pick   -> results/v6/v6_choice.json
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from phase2.p2lib import device, iso_from_oof, val_report  # noqa: E402
from v6.decode_v6 import decode_v6, record_source  # noqa: E402

RD = os.path.join(RESULTS, "v6")
V6, V6A, NAME, CODE = "xgb_v6_c2", "xgb_v6a_s2", "dmsA", 62
EPS_CAPS, EPS_EM, EPS_EM_C, EPS_G, EPS_PL, PL_PREC, EPS_PICK, TOL = 0.0001, 0.001, 0.0005, 0.001, 0.001, 0.99, 0.0005, 0.003


def jload(*p):
    path = os.path.join(*p)
    return json.load(open(path)) if os.path.exists(path) else None


def mn(r):
    return min(r["f0"], r["f4"])


def dms_scores(tag, sc, cty, dev, caps):
    """a system's OOF table on DMS: its own isotonic (folds 1-2) + its decode_eval gamma, optional caps."""
    path = os.path.join(CACHE, "feats", f"oof_{tag}.parquet")
    dj = jload(RESULTS, "xgboost", tag, "decode_eval.json")
    if not os.path.exists(path) or dj is None:
        return None
    D = pl.read_parquet(path)
    b = dj["expected_f"]["best"]
    keep, _, info = decode_v6(D, iso_from_oof(D), b["gamma"], b["extra"], dev, cty,
                              src=record_source("train") if caps else None)
    r = val_report(sc, keep, cty)
    return {"f0": r["f0"]["macro_f05"], "f4": r["f4"]["macro_f05"], "f3_es": r["f3_es"]["macro_f05"],
            "gamma": b["gamma"], "extra": b["extra"], "capped_records": info.get("capped_records", 0),
            "by_country_f0": {c: {"f05": v["macro_f05"], "precision": v["pair_precision"], "recall": v["pair_recall"],
                                  "fp": v["fp"], "fn": v["fn"], "pred_per_s1": v["pred_per_s1"]}
                              for c, v in r["f0"]["by_country"].items()}}


def loco_v6():
    return jload(RESULTS, "phase2", "loco_India2US_v6.json")


def cmd_plan():
    dev = device()
    cty = ids("train")[0]["country"].to_numpy()
    sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{CODE}.npy")))
    out = {"rule": __doc__}
    nocap, cap = dms_scores(V6, sc, cty, dev, False), dms_scores(V6, sc, cty, dev, True)
    assert nocap is not None, f"oof_{V6} / decode_eval missing: run the V6 collective rounds first"
    out["v6_dms"] = {"no_caps": nocap, "caps": cap}
    caps = mn(cap) >= mn(nocap) - EPS_CAPS
    ch = jload(RD, f"score_chain_{NAME}.json")
    em, em_det = False, None
    if ch and "universe_sim19_calibration_em" in ch:
        a, b = ch["universe_sim19_calibration"], ch["universe_sim19_calibration_em"]
        drops = {c: b["by_country_f0"][c]["f05"] - a["by_country_f0"][c]["f05"] for c in a["by_country_f0"]}
        em_det = {"gain_min_f0_f4": mn(b) - mn(a), "country_delta_f0": drops}
        em = em_det["gain_min_f0_f4"] > EPS_EM and min(drops.values()) >= -EPS_EM_C
    out["em_evidence"] = em_det
    L = loco_v6()
    g_seen = nocap["gamma"]
    gamma_unseen, g_det, pl_ok, pl_det = "same", None, False, None
    um = "US_all_unseen"
    if L and "stage1_all" in L["systems"]:
        s = L["systems"]["stage1_all"]
        gg = {float(k): v[um] for k, v in s.get("gamma_grid", {}).items()}
        if gg:
            g_best = max(gg, key=gg.get)
            g_near = min(gg, key=lambda g: abs(g - g_seen))
            g_det = {"grid": gg, "best": g_best, "nearest_to_seen": g_near, "gain": gg[g_best] - gg[g_near]}
            if g_det["gain"] > EPS_G:
                gamma_unseen = g_best
        if "stage1_pl" in L["systems"]:
            gain = L["systems"]["stage1_pl"]["iso+ef"][um]["macro_f05"] - s["iso+ef"][um]["macro_f05"]
            st = jload(RESULTS, "phase2", "loco_India2US_v6", "pl_step.json") or {}
            prec = st.get("pseudo_positive_precision")
            pl_det = {"gain_unseen": gain, "pseudo_positive_precision": prec, "pseudo_labels": st}
            pl_ok = gain > EPS_PL and (prec or 0) >= PL_PREC
    out.update({"gamma_evidence": g_det, "pl_evidence": pl_det,
                "plan": {"caps": bool(caps), "em": bool(em), "gamma_unseen": gamma_unseen, "pl": bool(pl_ok)}})
    os.makedirs(RD, exist_ok=True)
    json.dump(out, open(os.path.join(RD, "v6_plan.json"), "w"), indent=1, default=float)
    print(json.dumps(out["plan"]), flush=True)


def final_args():
    """the final_v6.py options of the plan (used by run_all.sh)."""
    p = jload(RD, "v6_plan.json")["plan"]
    a = (["--caps"] if p["caps"] else []) + (["--em"] if p["em"] else [])
    a += ["--gamma_unseen", str(p["gamma_unseen"])]
    print(" ".join(a))


def pl_on():
    """exit code 0 if the plan uses France pseudo-labels (run_all.sh gates the v6pl steps with it)."""
    p = jload(RD, "v6_plan.json")
    sys.exit(0 if p and p["plan"]["pl"] else 1)


def test_rate(out_name):
    d = jload(RESULTS, "final", out_name, "test_stats.json")
    if d is None:
        return None
    return {c: v.get("matches_per_s1", v["matches"] / max(v["s1"], 1)) for c, v in d["by_country"].items()}


def ratios(rate, val):
    return {c: rate[c] / max(val["by_country_f0"][c]["pred_per_s1"], 1e-9) for c in ("US", "India") if c in rate}


def cmd_pick():
    dev = device()
    cty = ids("train")[0]["country"].to_numpy()
    sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{CODE}.npy")))
    plan = jload(RD, "v6_plan.json")
    caps = bool(plan and plan["plan"]["caps"])
    ch = jload(RD, f"score_chain_{NAME}.json") or {}
    dms = {"V2_chain": ch.get("universe_sim19_calibration"), "V3noGNN_chain": ch.get("v3nognn_universe_sim19_calibration"),
           "V6a": dms_scores(V6A, sc, cty, dev, False), "V6": dms_scores(V6, sc, cty, dev, caps),
           "V6_round1": dms_scores("xgb_v6_c1", sc, cty, dev, caps)}
    out = {"rule": __doc__, "plan": plan["plan"] if plan else None, "ceiling_dms": ch.get("ceiling"),
           "dms_min_f0_f4": {k: (mn(v) if v else None) for k, v in dms.items()}, "dms": dms}
    ref = max([mn(dms[k]) for k in ("V2_chain", "V3noGNN_chain") if dms[k]] or [0.0])
    v2rate, v2val = test_rate("final_v2"), dms["V2_chain"]
    r_v2 = ratios(v2rate, v2val) if v2rate and v2val else None
    out["v2_test_over_dms_ratio"] = r_v2
    checks = {}
    for sysname, files in (("V6", ["final_v6pl", "final_v6"] if plan and plan["plan"]["pl"] else ["final_v6"]),
                           ("V6a", ["final_v6a"])):
        d = dms[sysname]
        for f in files:
            rate = test_rate(f)
            if d is None or rate is None:
                checks[f] = {"available": False}
                continue
            r = ratios(rate, d)
            cons = r_v2 is None or all(r[c] <= r_v2[c] + TOL for c in r)
            better = mn(d) >= ref + EPS_PICK
            checks[f] = {"available": True, "dms_min_f0_f4": mn(d), "reference": ref, "better": better,
                         "test_over_dms_ratio": r, "consistent": cons, "pass": bool(better and cons)}
    out["checks"] = checks
    order = [f for f, c in checks.items() if c.get("pass")]
    if test_rate("final_v3nognn") is not None:
        order.append("final_v3nognn")
    order.append("final_v2 (keep)")
    out["submit_order"] = order
    out["first_choice_file"] = (f"results/final/{order[0]}/output/matching_results.tsv" if order[0] != "final_v2 (keep)"
                                else "submissions/final_v2/matching_results.tsv")
    os.makedirs(RD, exist_ok=True)
    json.dump(out, open(os.path.join(RD, "v6_choice.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rule", "dms")}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    {"plan": cmd_plan, "pick": cmd_pick, "final_args": final_args, "pl_on": pl_on}[sys.argv[1] if len(sys.argv) > 1 else "pick"]()
