"""Pre-registered V4 selection rule (written before any phase-2 result exists; do not tune it on the results).

Inputs: results/phase2/ablation.json (+ loco_*.json when present).
For each candidate in {V2, V4A, V4B, V4C} with scores on validation and test:
  R  = min(F0.5 fold 0, F0.5 fold 4)                              robust held-out score (both folds clean for chain A)
  Gate 1 (test/validation consistency, label-free; V3 fails it):
     for US and India: drift = (test/val predicted matches per S1) - (same ratio for V2); |drift| <= 0.004
     and uncertain share of assigned records, test / validation fold 0 <= 1.5
  Gate 2 (unseen-country generalisation, LOCO India -> US, iso+EF decoder; leakage-free run preferred when present):
     a component may only be used if adding it does not hurt the unseen country by more than 0.001 F0.5:
     GNN  : gnn_no_xlmr - stage1 on US_f0_unseen >= -0.001          (V4A, V4B)
     XLM-R: gnn+xlmr - gnn_no_xlmr on US_f0_unseen >= -0.001        (V4A)
  Choice: highest R among candidates passing both gates; a candidate must beat V2's R by more than 0.0003 or V2-like
  simplicity wins (order of preference on ties: V4C, V4B, V4A). If none passes, V2 stays the submission.

  python code/phase2/select_v4.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import RESULTS  # noqa: E402
from phase2.p2lib import P2, dump  # noqa: E402

DRIFT_MAX, UNC_RATIO_MAX, LOCO_TOL, MIN_GAIN = 0.004, 1.5, 0.001, 0.0003
SIMPLICITY = ["V2", "V4C", "V4B", "V4A"]
FINAL_CMD = {
    "V4A": "python -u code/final/final_from_scores.py --test_scores p2/gnn/v4a_chainA_hash1_test_scores.parquet --col gnn "
           "--logit --oof oof_gnn_mlx_v2.parquet --decode results/gnn/gnn_mlx_v2/decode_eval.json --out final_v4",
    "V4B": "python -u code/final/final_from_scores.py --test_scores p2/gnn/v4b_gnnA_hash1_test_scores.parquet --col gnn "
           "--logit --oof oof_gnn_v2.parquet --decode results/gnn/gnn_v2/decode_eval.json --out final_v4",
    "V4C": "python -u code/final/final_from_scores.py --test_scores feats/p2_test_v4c_hash1_scores.parquet --col p "
           "--oof oof_xgb_c2_blend_s2_v2.parquet --decode results/xgboost/xgb_c2_blend_s2_v2/decode_eval.json --out final_v4",
}


def loco_gates():
    for name in ("loco_India2US_noemb.json", "loco_India2US.json"):
        p = os.path.join(P2, name)
        if os.path.exists(p):
            d = json.load(open(p))["component_deltas_iso_ef"]
            g = d.get("gnn_no_xlmr - stage1", {}).get("US_f0_unseen")
            x = d.get("gnn+xlmr - gnn_no_xlmr", {}).get("US_f0_unseen")
            return {"source": name, "gnn_delta_unseen": g, "xlmr_delta_unseen": x,
                    "gnn_ok": g is None or g >= -LOCO_TOL, "xlmr_ok": x is None or x >= -LOCO_TOL}
    return {"source": None, "gnn_ok": True, "xlmr_ok": True, "note": "no LOCO result: gate 2 not applied"}


def main():
    ab = json.load(open(os.path.join(P2, "ablation.json")))["systems"]
    lg = loco_gates()
    v2 = ab["V2"]
    v2_ratio = {c: v2["consistency"][c]["ratio_test_over_val"] for c in ("US", "India")}
    rows = {}
    for name in SIMPLICITY:
        s = ab.get(name)
        if not s or s.get("status") != "ok":
            rows[name] = {"status": "missing"}
            continue
        R = min(s["val"]["f0"]["macro_f05"], s["val"]["f4"]["macro_f05"])
        g1, why = True, []
        for c in ("US", "India"):
            cons = s["consistency"][c]
            drift = cons["ratio_test_over_val"] - v2_ratio[c]
            if abs(drift) > DRIFT_MAX:
                g1 = False
                why.append(f"{c} drift {drift:+.4f}")
            if cons.get("uncertain_share_val_f0") and cons.get("uncertain_share_test") is not None:
                ur = cons["uncertain_share_test"] / cons["uncertain_share_val_f0"]
                if ur > UNC_RATIO_MAX:
                    g1 = False
                    why.append(f"{c} uncertain share x{ur:.2f}")
        g2 = {"V4A": lg["gnn_ok"] and lg["xlmr_ok"], "V4B": lg["gnn_ok"]}.get(name, True)
        if not g2:
            why.append("LOCO: component hurts the unseen country")
        rows[name] = {"status": "ok", "R": R, "f0": s["val"]["f0"]["macro_f05"], "f4": s["val"]["f4"]["macro_f05"],
                      "gate_consistency": g1, "gate_loco": g2, "why_rejected": why,
                      "test_matches": s["test"]["matches"], "delta_vs_V2": s["test"].get("delta_vs_ref")}
    ok = [n for n in SIMPLICITY if rows[n].get("status") == "ok" and rows[n]["gate_consistency"] and rows[n]["gate_loco"]]
    choice, R2 = "V2", rows["V2"]["R"]
    for n in ok:
        if n != "V2" and rows[n]["R"] > max(R2 + MIN_GAIN, rows[choice]["R"] + (MIN_GAIN if choice != "V2" else 0)):
            choice = n
    out = {"rule": __doc__, "loco": lg, "candidates": rows, "choice": choice,
           "final_command": FINAL_CMD.get(choice, "V2 stays: submissions/final_v2/matching_results.tsv (no V4 file written)")}
    dump(out, "v4_choice.json")
    print(json.dumps({k: out[k] for k in ("choice", "final_command")}, indent=1))
    for n, r in rows.items():
        print(n, r)


if __name__ == "__main__":
    main()
