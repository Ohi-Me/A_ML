"""Which V5 file to submit (rule fixed before any V5 result exists; do not tune it on the results).

Evidence used (no test labels):
  val      : macro F0.5 on SIM19 fold 0 and fold 4 of V2 (xgb_c2_blend_s2_v2) and V5 (xgb_v5_s2), iso + EF decoding
  LOCO     : India -> US, stage 1 refit on India folds 1-4 (stage1_all), with the V2 features (loco_..._v2) and the V5
             features (loco_..._v5); and the pseudo-label gain stage1_pl - stage1_all on the unseen country
Rule
  1. V5 features are used only if min(f0, f4) of V5 >= that of V2 - 0.0002 (not worse in-domain) AND the V5 features
     are not worse on the unseen country in LOCO (US_all_unseen, iso+ef) by more than 0.0005.
  2. Pseudo-labels (final_v5pl) only if, in LOCO v5, stage1_pl beats stage1_all on US_all_unseen by > 0.001 and the
     pseudo-positive precision there is >= 0.99.
  3. Submit in this order: final_v5pl (1 and 2 pass), final_v5 (1 passes), otherwise V2 stays.

  python code/v5/choose_v5.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import RESULTS  # noqa: E402

EPS_VAL, EPS_LOCO, MIN_PL = 0.0002, 0.0005, 0.001


def val_scores(tag, sc, dev):
    """macro F0.5 on the clean folds 0 and 4 (+ per country), isotonic folds 1-2 + expected-F0.5 of that run."""
    import polars as pl
    from common.decide import ids
    from common.io import CACHE
    from phase2.p2lib import decode, iso_from_oof, val_report
    D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{tag}.parquet"))
    b = json.load(open(os.path.join(RESULTS, "xgboost", tag, "decode_eval.json")))["expected_f"]["best"]
    keep, _ = decode(D.select(["a", "b", "p"]), rule="ef", iso=iso_from_oof(D), gamma=b["gamma"], extra=b["extra"], dev=dev, M=4096)
    r = val_report(sc, keep, ids("train")[0]["country"].to_numpy())
    return {"f0": r["f0"]["macro_f05"], "f4": r["f4"]["macro_f05"],
            "by_country_f0": {c: v["macro_f05"] for c, v in r["f0"]["by_country"].items()}}


def loco(name):
    p = os.path.join(RESULTS, "phase2", f"loco_India2US_{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    import numpy as np
    from common.decide import Scorer
    from common.io import CACHE
    from phase2.p2lib import device
    dev = device()
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    out = {"rule": __doc__}
    v2, v5 = val_scores("xgb_c2_blend_s2_v2", sc, dev), val_scores("xgb_v5_s2", sc, dev)
    out["val"] = {"V2": v2, "V5": v5}
    r2, r5 = min(v2["f0"], v2["f4"]), min(v5["f0"], v5["f4"])
    L2, L5 = loco("v2"), loco("v5")
    um = "US_all_unseen"
    l2 = L2["systems"]["stage1_all"]["iso+ef"][um]["macro_f05"] if L2 and "stage1_all" in L2["systems"] else None
    l5 = L5["systems"]["stage1_all"]["iso+ef"][um]["macro_f05"] if L5 and "stage1_all" in L5["systems"] else None
    out["loco_unseen"] = {"v2_features": l2, "v5_features": l5}
    feat_ok = r5 >= r2 - EPS_VAL and (l2 is None or l5 is None or l5 >= l2 - EPS_LOCO)
    pl_gain, pl_prec = None, None
    if L5 and "stage1_pl" in L5["systems"] and "stage1_all" in L5["systems"]:
        pl_gain = L5["systems"]["stage1_pl"]["iso+ef"][um]["macro_f05"] - l5
        stp = os.path.join(RESULTS, "phase2", "loco_India2US_v5", "pl_step.json")
        pl_prec = json.load(open(stp)).get("pseudo_positive_precision") if os.path.exists(stp) else None
    pl_ok = feat_ok and pl_gain is not None and pl_gain > MIN_PL and (pl_prec or 0) >= 0.99
    out.update({"features_ok": feat_ok, "pl_gain_unseen": pl_gain, "pl_positive_precision": pl_prec, "pl_ok": pl_ok})
    order = (["final_v5pl"] if pl_ok else []) + (["final_v5"] if feat_ok else []) + ["final_v2 (keep)"]
    out["submit_order"] = order
    out["first_choice_file"] = (f"results/final/{order[0]}/output/matching_results.tsv" if order[0] != "final_v2 (keep)"
                                else "submissions/final_v2/matching_results.tsv")
    os.makedirs(os.path.join(RESULTS, "v5"), exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "v5", "v5_choice.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()
