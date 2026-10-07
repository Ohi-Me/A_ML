"""Ablation matrix on one candidate population (the C2 v2 candidates): which V3 component causes the test regression?

For every system: validation (SIM19 test-prior simulation) on the clean held-out folds 0 and 4 (+ fold 3, which was
the XGBoost early-stopping fold), per country, bootstrap CI on fold 0; and label-free test decisions: matches, empty
(singleton) rate, per country, delta vs V2, and how many of the V3-only accepts (V3 \\ V2) survive.
Consistency diagnostics (label-free, the V3 symptom): test/validation ratio of predicted matches per S1 and of the
uncertain share of assigned records, per country.

Systems whose score files are missing are listed as "missing" (their job has not run yet). Decisions are cached in
p2/decisions/, so re-running only computes what is new.

  python code/phase2/ablation.py                       # all known systems
  python code/phase2/ablation.py --only V2,V3 --no_boot
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.decode import assign  # noqa: E402
from phase2.p2lib import (calibrate, cpath, decode, device, dump, exists, iso_from_oof, load_scores, logit,  # noqa: E402
                          test_report, tune_threshold, uncertain_share, val_report)

DDIR = os.path.join(CACHE, "p2", "decisions")
XTR, XTE = "xenc/ml_hardv2_train_scores.parquet", "xenc/ml_hardv2_test_scores.parquet"


def dec_settings(path):
    d = json.load(open(os.path.join(RESULTS, path)))
    b = d["expected_f"]["best"]
    return {"gamma": b["gamma"], "extra": b["extra"]}


V2_DEC = "xgboost/xgb_c2_blend_s2_v2/decode_eval.json"
V3_DEC = "gnn/gnn_mlx_v2/decode_eval.json"

# name -> (description, val OOF under feats/, test (path, col, is_logit), decoder)
#   decoder: rule ef|thr, iso True|False, dec: decode_eval.json whose gamma/extra is used
SYSTEMS = {
    "V2": ("V2 baseline: stage-1 blend -> XGB stage 2 -> isotonic -> expected-F0.5",
           "oof_xgb_c2_blend_s2_v2.parquet", ("feats/test_scores_final_v2.parquet", "p", False), {"rule": "ef", "iso": True, "dec": V2_DEC}),
    "V3": ("V3 full: stage-1 blend -> XLM-R (band) -> GNN -> isotonic -> expected-F0.5",
           "oof_gnn_mlx_v2.parquet", ("gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_noGNN": ("V3 without GNN: logistic stack of stage-1 blend + XLM-R (band pairs)",
                 "@stack", ("@stack", None, False), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_noXLMR": ("V3 without XLM-R = GNN without the XLM-R edge input (gnn_v2); in V3 XLM-R only enters as a GNN "
                  "edge input, so ablations 4 and 5 are the same model",
                  "oof_gnn_v2.parquet", ("gnn/gnn_v2_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": "gnn/gnn_v2/decode_eval.json"}),
    "V3_GNNnoCos": ("GNN(+XLM-R) retrained without the bi-encoder cosine edge input",
                    "oof_p2_gnn_mlx_nocos.parquet", ("p2/gnn/gnn_mlx_nocos_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_cosHash1_GNN": ("V3 checkpoints, test cosine computed the validation way in the GNN edges only "
                        "(stage 1 and XLM-R unchanged); validation is V3's own",
                        "oof_gnn_mlx_v2.parquet", ("p2/gnn/v3_gnnhash1_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_cosHash1_chain": ("V3 checkpoints, validation-style cosine everywhere on test: stage 1 refit rescored, XLM-R band "
                          "re-selected and scored, GNN edges",
                          "oof_gnn_mlx_v2.parquet", ("p2/gnn/v3_chainhash1_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_V2decoder": ("V3 scores with V2's decoder (isotonic folds 1-2 + expected-F0.5 with V2's gamma/extra)",
                     "oof_gnn_mlx_v2.parquet", ("gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V2_DEC}),
    "V3_noIso": ("V3 without isotonic calibration (expected-F0.5 on the raw GNN probability)",
                 "oof_gnn_mlx_v2.parquet", ("gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": False, "dec": V3_DEC}),
    "V3_thrMargin": ("V3 with a simple threshold + margin decoder (tuned on folds 1-2)",
                     "oof_gnn_mlx_v2.parquet", ("gnn/gnn_mlx_v2_test_scores.parquet", "gnn", True), {"rule": "thr", "iso": False, "dec": V3_DEC}),
    "V3_stage1halfA": ("V3 GNN fed test stage-1 scores of half model A (the kind it was trained on) instead of the refit "
                       "(gap_audit job gnn_cfA)",
                       "oof_gnn_mlx_v2.parquet", ("gnn/gnn_cfA_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V3_stage1halfB": ("same with half model B (gnn_cfB)",
                       "oof_gnn_mlx_v2.parquet", ("gnn/gnn_cfB_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V4A": ("V4A corrected V3 = V3 chain A replayed on test (stage-1 half A, XLM-R A on A's own band, GNN A) with the "
            "validation-style cosine: the test score is produced exactly the way fold 0 was",
            "oof_gnn_mlx_v2.parquet", ("p2/gnn/v4a_chainA_hash1_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": V3_DEC}),
    "V4B": ("V4B simplified: chain A replay without XLM-R (stage-1 half A -> GNN gnn_v2 A), validation-style cosine",
            "oof_gnn_v2.parquet", ("p2/gnn/v4b_gnnA_hash1_test_scores.parquet", "gnn", True), {"rule": "ef", "iso": True, "dec": "gnn/gnn_v2/decode_eval.json"}),
    "V4C": ("V4C V2-style: chain A replay of V2 (stage-1 half A -> stage-2 XGB model A), validation-style cosine",
            "oof_xgb_c2_blend_s2_v2.parquet", ("feats/p2_test_v4c_hash1_scores.parquet", "p", False), {"rule": "ef", "iso": True, "dec": V2_DEC}),
}


def stack_tables(dev):
    """V3 without GNN: logistic stack [logit p, x, has, logit p * x] fitted on folds 1-2 (stack_eval.py design)."""
    from sklearn.linear_model import LogisticRegression

    def design(p, x):
        has = ~np.isnan(x)
        x0 = np.where(has, x, 0.0)
        return np.column_stack([logit(p), x0, has.astype(float), logit(p) * x0]), has
    D = pl.read_parquet(cpath("feats/oof_xgb_c2_blend_v2.parquet")).join(pl.read_parquet(cpath(XTR)).select(["a", "b", "mlxenc"]),
                                                                          on=["a", "b"], how="left")
    Z, h = design(D["p"].to_numpy(), D["mlxenc"].to_numpy().astype(np.float64))
    tr = h & np.isin(D["fold"].to_numpy(), [1, 2])
    lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], D["y"].to_numpy()[tr])
    p = D["p"].to_numpy().astype(np.float64)
    p[h] = lr.predict_proba(Z[h])[:, 1]
    V = D.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p.astype(np.float32)))
    T = pl.read_parquet(cpath("feats/test_stage1_final_v2.parquet")).join(pl.read_parquet(cpath(XTE)).select(["a", "b", "mlxenc"]),
                                                                           on=["a", "b"], how="left")
    Z, h = design(T["p"].to_numpy(), T["mlxenc"].to_numpy().astype(np.float64))
    p = T["p"].to_numpy().astype(np.float64)
    p[h] = lr.predict_proba(Z[h])[:, 1]
    return V, T.select(["a", "b"]).with_columns(pl.Series("p", p.astype(np.float32)))


def submission_pairs(name):
    """(a, b) of a submitted matching_results.tsv, to check that the replay reproduces it."""
    p = os.path.join(os.getcwd(), "submissions", name, "matching_results.tsv")
    if not os.path.exists(p):
        return None
    d = pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False).rename(
        {"source1_entity_id": "s1", "matched_entity_ids": "m"}).filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
    d = d.with_columns(pl.col("m").str.split(",")).explode("m")
    s1, rr = ids("test")
    m1 = pl.DataFrame({"s1": s1["id"], "a": np.arange(s1.height, dtype=np.int32)})
    mr = pl.DataFrame({"m": rr["id"], "b": np.arange(rr.height, dtype=np.int32)})
    return d.join(m1, on="s1").join(mr, on="m").select(["a", "b"])


def run_system(name, spec, sc, c_tr, c_te, dev, n_boot, force, stack_cache):
    desc, voof, (tpath, tcol, tlog), dcfg = spec
    vfile, tfile = os.path.join(DDIR, f"{name}_val.parquet"), os.path.join(DDIR, f"{name}_test.parquet")
    if voof == "@stack":
        if "t" not in stack_cache:
            stack_cache["t"] = stack_tables(dev)
        V, T = stack_cache["t"]
    else:
        if not exists(f"feats/{voof}") or not exists(tpath):
            return {"desc": desc, "status": "missing", "need": [x for x in (f"feats/{voof}", tpath) if not exists(x)]}
        V = pl.read_parquet(cpath(f"feats/{voof}"))
        T = load_scores(tpath, tcol, tlog)
    st = dec_settings(dcfg["dec"])
    iso = iso_from_oof(V) if dcfg["iso"] else None
    kw = dict(rule=dcfg["rule"], iso=iso, gamma=st["gamma"], extra=st["extra"], dev=dev, M=4096)
    if dcfg["rule"] == "thr":
        tt = tune_threshold(V.select(["a", "b", "p"]), sc)
        kw.update(thr=tt["thr"], margin=tt["margin"])
        st.update(tt)
    t0 = time.time()
    if os.path.exists(vfile) and not force:
        kv, Av = pl.read_parquet(vfile), None
    else:
        kv, Av = decode(V.select(["a", "b", "p"]), **kw)
        kv.write_parquet(vfile)
    if os.path.exists(tfile) and not force:
        kt, At = pl.read_parquet(tfile), None
    else:
        kt, At = decode(T, **kw)
        kt.write_parquet(tfile)
    r = {"desc": desc, "status": "ok", "decoder": {**dcfg, **st}, "val": val_report(sc, kv, c_tr, n_boot)}
    if Av is None:                                    # cached decisions: rebuild the (cheap) assignment table
        Av = assign(calibrate(V.select(["a", "b", "p"]), iso), floor=0.01)
    if At is None:
        At = assign(calibrate(T, iso), floor=0.01)
    if Av is not None:
        r["val_uncertain_share_f0"] = uncertain_share(Av.filter(pl.Series(sc.fold[Av["a"].to_numpy()] == 0)), c_tr)
    if At is not None:
        r["test_uncertain_share"] = uncertain_share(At, c_te)
    r["_keep_test"] = kt
    r["_keep_val"] = kv
    r["runtime_s"] = time.time() - t0
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--extra_systems", default="", help="json file: {name: [desc, val_oof, [test_path, col, is_logit], decoder]}")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--no_boot", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="ablation.json")
    args = ap.parse_args()
    dev = device()
    os.makedirs(DDIR, exist_ok=True)
    systems = dict(SYSTEMS)
    if args.extra_systems:
        for k, v in json.load(open(args.extra_systems)).items():
            systems[k] = (v[0], v[1], tuple(v[2]), v[3])
    names = [x for x in args.only.split(",") if x] or list(systems)
    for base in ("V3", "V2"):
        if base not in names:
            names.insert(0, base)
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    c_tr, c_te = ids("train")[0]["country"].to_numpy(), ids("test")[0]["country"].to_numpy()
    res, stack_cache = {}, {}
    for n in names:
        print(f"== {n}", flush=True)
        res[n] = run_system(n, systems[n], sc, c_tr, c_te, dev, 0 if args.no_boot else args.n_boot, args.force, stack_cache)
        print(n, res[n]["status"], res[n].get("val", {}).get("f0", {}).get("macro_f05"), flush=True)
    v2t, v3t = res["V2"]["_keep_test"], res["V3"]["_keep_test"]
    v2v, v3v = res["V2"]["_keep_val"], res["V3"]["_keep_val"]
    extra_t = v3t.join(v2t, on=["a", "b"], how="anti")            # the V3-only test accepts
    extra_v = v3v.join(v2v, on=["a", "b"], how="anti")
    val0 = sc.fold == 0
    out = {"extra_accepts_test": {"total": extra_t.height,
                                  "by_country": {c: int((c_te[extra_t["a"].to_numpy()] == c).sum()) for c in sorted(set(c_te.tolist()))}},
           "extra_accepts_val_f0": {"total": int(val0[extra_v["a"].to_numpy()].sum())},
           "systems": {}}
    for sub, name in (("V2", "final_v2"), ("V3", "final_v3")):
        sp = submission_pairs(name)
        if sp is not None:
            k = res[sub]["_keep_test"]
            both = k.join(sp, on=["a", "b"], how="semi").height
            out[f"replay_check_{sub}"] = {"replay": k.height, "submitted": sp.height, "common": both}
    n_te = {c: int((c_te == c).sum()) for c in set(c_te.tolist())}
    for n, r in res.items():
        if r["status"] != "ok":
            out["systems"][n] = r
            continue
        kt, kv = r.pop("_keep_test"), r.pop("_keep_val")
        r["test"] = test_report(kt, c_te, extra_t, v2t)
        # consistency: how does the change vs V2 on test compare with what validation (fold 0) predicted?
        cons = {}
        for c, d in r["test"]["by_country"].items():
            vb = r["val"]["f0"]["by_country"].get(c)
            vb2 = res["V2"]["val"]["f0"]["by_country"].get(c) if "val" in res["V2"] else None
            e = {"test_pred_per_s1": d["pred_per_s1"]}
            if vb:
                e["val_pred_per_s1"] = vb["pred_per_s1"]
                e["ratio_test_over_val"] = d["pred_per_s1"] / max(vb["pred_per_s1"], 1e-9)
                if vb2 and n != "V2":
                    dv = vb["pred_per_s1"] - vb2["pred_per_s1"]
                    dt = d["delta_vs_ref"] / n_te[c]
                    e["delta_vs_V2_per_s1_val"], e["delta_vs_V2_per_s1_test"] = dv, dt
                    e["inflation_test_over_val"] = dt / dv if abs(dv) > 1e-4 else None
            if "test_uncertain_share" in r and "val_uncertain_share_f0" in r and c in r["val_uncertain_share_f0"]:
                e["uncertain_share_val_f0"] = r["val_uncertain_share_f0"][c]
                e["uncertain_share_test"] = r["test_uncertain_share"].get(c)
            cons[c] = e
        r["consistency"] = cons
        # V3-only accepts on validation: how many survive in this system, and how precise they were
        if n != "V2":
            ev = extra_v.filter(pl.Series(val0[extra_v["a"].to_numpy()]))
            kept = ev.join(kv, on=["a", "b"], how="semi")
            r["val_extra_f0"] = {"extra": ev.height, "kept": kept.height,
                                 "precision_of_kept": float((sc.true_s1_of_b[kept["b"].to_numpy()] == kept["a"].to_numpy()).mean())
                                 if kept.height else None}
        out["systems"][n] = r
    # compact table
    rows = []
    for n, r in out["systems"].items():
        if r.get("status") != "ok":
            rows.append({"system": n, "status": r["status"]})
            continue
        row = {"system": n, "val_f0": r["val"]["f0"]["macro_f05"], "val_f4": r["val"]["f4"]["macro_f05"],
               "val_f3_es": r["val"]["f3_es"]["macro_f05"], "ci95_f0": str(r["val"]["f0"].get("ci95")),
               "test_matches": r["test"]["matches"], "test_empty_rate": r["test"]["empty_rate"],
               "delta_vs_V2": r["test"].get("delta_vs_ref"), "extra_removed": r["test"].get("extra_removed")}
        for c, d in r["test"]["by_country"].items():
            row[f"test_{c}_matches"] = d["matches"]
            row[f"test_{c}_delta_vs_V2"] = d.get("delta_vs_ref")
            row[f"test_{c}_extra_removed"] = d.get("extra_removed")
        for c, d in r["val"]["f0"]["by_country"].items():
            row[f"val_f0_{c}"] = d["macro_f05"]
        for c, d in r["consistency"].items():
            row[f"inflation_{c}"] = d.get("inflation_test_over_val")
        rows.append(row)
    out["table"] = rows
    dump(out, args.out)
    tab = pl.DataFrame(rows, infer_schema_length=None)
    tab.write_csv(os.path.join(RESULTS, "phase2", args.out.replace(".json", ".csv")))
    with pl.Config(tbl_cols=12, tbl_rows=40):
        print(tab.select([c for c in ("system", "val_f0", "val_f4", "test_matches", "delta_vs_V2", "extra_removed",
                                      "inflation_US", "inflation_India") if c in tab.columns]), flush=True)


if __name__ == "__main__":
    main()
