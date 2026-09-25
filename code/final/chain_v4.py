"""V4: run the test set through exactly the chains that were validated, instead of a refit.

What went wrong in V3: every downstream model (cross-encoder band, GNN, calibration, decoder) was trained and tuned on
stage-1 scores of half models (each fitted on 40% of the S1), but on test it was fed the scores of a stage-1 model
refitted on 100% of the data with 1.25x the rounds - sharper scores the GNN had never seen.

Chains (A: models fitted on folds {1,2}; B: models fitted on folds {3,4}; both never saw fold 0):
  stage 1 (full + no-embedding blend) -> uncertain band -> cross-encoder of the same half -> GNN of the same half.
The test score is the mean of the chain A and chain B GNN logits. Validation fold 0 is split by S1 into a calibration
half (isotonic, rule and its settings) and an evaluation half (reported), so the ensemble is calibrated on scores of
its own kind. "R" replays what V3 did on test (all-folds-but-0 refit stage 1 -> mean-of-halves cross-encoder / GNN).

Steps (each a separate job):
  python code/final/chain_v4.py refit        # R: stage-1 full / no-embedding fitted on folds 1-4 (x1.25 rounds)
  python code/final/chain_v4.py p1           # fold-0 stage-1 scores pA (OOF) / pB / pR, test pA / pB
  (cross-encoder)  xenc_score_pairs.py on feats/train_f0_p1.parquet and feats/test_chain_p1.parquet
  python code/final/chain_v4.py extras       # GNN input tables per chain
  (GNN)            edge_gnn.py --score_only ... per chain
  python code/final/chain_v4.py evaluate     # chain A / B / ensemble / R on the evaluation half of fold 0
  python code/final/chain_v4.py final --system ens --out final_v4
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record, decide_records, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, write_id_lists  # noqa: E402
from common.xgbdata import PartIter  # noqa: E402

RDIR = os.path.join(RESULTS, "final", "chain_v4")
FEATS = "c2_train_sim19"
BLEND = "xgb_c2_blend_v2"
OOF_S1 = "oof_xgb_c2_blend_v2.parquet"
OOF_GNN = "oof_gnn_mlx_v2.parquet"
XENC_TRAIN = "xenc/ml_hardv2_train_scores.parquet"


def blend_info():
    bl = json.load(open(os.path.join(RESULTS, "xgboost", BLEND, "report.json")))["blend"]
    return bl, {t: json.load(open(os.path.join(RESULTS, "xgboost", t, "report.json"))) for t in (bl["full"], bl["noemb"])}


def country_w(split, bl):
    seen = set(ids("train")[0]["country"].unique().to_list())
    cty = ids(split)[0]["country"].to_numpy()
    return np.where(np.isin(cty, list(seen)), bl["w_seen"], bl["w_unseen"]).astype(np.float32)


def cmd_refit(args):
    bl, reps = blend_info()
    files = sorted(glob.glob(os.path.join(CACHE, "feats", FEATS, "part_*.parquet")))
    os.makedirs(RDIR, exist_ok=True)
    for tag, rp in reps.items():
        out = os.path.join(RDIR, f"stage1_{tag}_f1234.json")
        if os.path.exists(out):
            continue
        with Timer(f"refit {tag} on folds 1-4, {int(rp['rounds'] * 1.25)} rounds"):
            dtr = xgb.QuantileDMatrix(PartIter(files, rp["features"], [1, 2, 3, 4]), max_bin=256)
            m = xgb.train(rp["params"], dtr, int(rp["rounds"] * 1.25))
            m.save_model(out)
            del dtr, m


def cmd_p1(args):
    """fold-0 pairs of the training table with pA (the OOF score), pB (half B), pR (refit 1-4); test pA / pB."""
    bl, reps = blend_info()
    w_tr = country_w("train", bl)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", FEATS, "part_*.parquet")))
    oof = pl.read_parquet(os.path.join(CACHE, "feats", OOF_S1))
    mods = {}
    for tag, rp in reps.items():
        mods[tag] = {"B": xgb.Booster(model_file=os.path.join(RESULTS, "xgboost", tag, "model_B.json"))}
        r = os.path.join(RDIR, f"stage1_{tag}_f1234.json")
        if os.path.exists(r):
            mods[tag]["R"] = xgb.Booster(model_file=r)
    kinds = [k for k in ("B", "R") if all(k in mods[t] for t in mods)]
    with Timer(f"fold-0 stage-1 scores {kinds}"):
        cols = {k: np.full(oof.height, np.nan, np.float32) for k in kinds}
        off = 0
        for f in files:
            d = pl.read_parquet(f)
            n = d.height
            v = d["fold"].to_numpy() == 0
            if v.any():
                dv = d.filter(pl.Series(v))
                ww = w_tr[dv["a"].to_numpy()]
                for k in kinds:
                    ps = [mods[t][k].predict(xgb.DMatrix(dv.select([pl.col(c).cast(pl.Float32) for c in reps[t]["features"]]).to_numpy()))
                          for t in (bl["full"], bl["noemb"])]
                    cols[k][off:off + n][v] = ww * ps[0] + (1 - ww) * ps[1]
            off += n
        assert off == oof.height
        T = oof.with_columns([pl.Series(f"p{k}", cols[k]) for k in kinds]).rename({"p": "pA"}).filter(pl.col("fold") == 0)
        T.write_parquet(os.path.join(CACHE, "feats", "train_f0_p1.parquet"))
        print("fold-0 pairs", T.height, {c: float(T[c].mean()) for c in T.columns if c.startswith("p")}, flush=True)
    with Timer("test pA / pB"):
        A = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_halfA_v2.parquet"))
        B = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_halfB_v2.parquet"))
        assert (A["b"].to_numpy() == B["b"].to_numpy()).all()
        A.rename({"p": "pA"}).with_columns(B["p"].alias("pB")).write_parquet(os.path.join(CACHE, "feats", "test_chain_p1.parquet"))


def cmd_extras(args):
    """GNN input tables: stage-1 score tables aligned with the feature parts, cross-encoder tables (a, b, mlxenc)."""
    oof = pl.read_parquet(os.path.join(CACHE, "feats", OOF_S1))
    f0 = pl.read_parquet(os.path.join(CACHE, "feats", "train_f0_p1.parquet"))
    xo = pl.read_parquet(os.path.join(CACHE, XENC_TRAIN))
    xo_f = xo.join(oof.select(["a", "b", "fold"]), on=["a", "b"], how="left")
    rest = xo_f.filter(pl.col("fold") != 0).select(["a", "b", "mlxenc"])
    xf0 = pl.read_parquet(os.path.join(CACHE, "xenc", "train_f0_chain.parquet"))
    v = (oof["fold"] == 0).to_numpy()
    for k in [c[1:] for c in f0.columns if c in ("pB", "pR")]:
        p = oof["p"].to_numpy().copy()
        p[v] = f0[f"p{k}"].to_numpy()
        oof.with_columns(pl.Series("p", p)).write_parquet(os.path.join(CACHE, "feats", f"oof_chain{k}_s1.parquet"))
        if k == "B":
            x0 = xf0.filter(pl.col("xB").is_not_null()).select(["a", "b", pl.col("xB").alias("mlxenc")])
        else:
            x0 = xf0.filter(pl.col("xRA").is_not_null()).select(["a", "b", ((pl.col("xRA") + pl.col("xRB")) / 2).alias("mlxenc")])
        pl.concat([rest, x0]).write_parquet(os.path.join(CACHE, "xenc", f"chain{k}_train_extra.parquet"))
        print("chain", k, "fold-0 cross-encoder pairs", x0.height, flush=True)
    xt = pl.read_parquet(os.path.join(CACHE, "xenc", "chain_test.parquet"))
    for k in ("A", "B"):
        x = xt.filter(pl.col(f"x{k}").is_not_null()).select(["a", "b", pl.col(f"x{k}").alias("mlxenc")])
        x.write_parquet(os.path.join(CACHE, "xenc", f"chain{k}_test_extra.parquet"))
        print("test chain", k, "cross-encoder pairs", x.height, flush=True)


# ---------------------------------------------------------------- evaluation
def half_of(s1_ids):
    return np.array([zlib.crc32((x + "|cal").encode()) % 2 for x in s1_ids], np.int8)


def fit_iso(p, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def sig(x):
    return 1 / (1 + np.exp(-x))


def choose_rule(D, sc, cal_mask, eval_mask):
    """D: a, b, p (calibrated) over all edges. Rule and settings chosen on cal_mask, reported on eval_mask."""
    res = {}
    best = best_per_record(D)
    grid = []
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        for mg in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            m, _ = sc.metrics(decide_records(best, thr, mg), cal_mask)
            grid.append((m["macro_f05"], thr, mg))
    f, thr, mg = max(grid)
    keep_t = decide_records(best, thr, mg)
    res["threshold"] = {"thr": thr, "margin": mg, "f05_cal": f, "eval": sc.metrics(keep_t, eval_mask)[0]}
    A = assign(D, floor=0.01)
    g = []
    for gamma in (0.8, 1.0, 1.25, 1.5):
        keep, _ = expected_f_decode(A, gamma=gamma, M=1024)
        g.append((sc.metrics(keep, cal_mask)[0]["macro_f05"], gamma))
    f2, gamma = max(g)
    keep_e, _ = expected_f_decode(A, gamma=gamma, M=4096)
    res["expected_f"] = {"gamma": gamma, "f05_cal": f2, "eval": sc.metrics(keep_e, eval_mask)[0]}
    res["chosen"] = "expected_f" if f2 >= f else "threshold"
    res["eval_f05"] = res[res["chosen"]]["eval"]["macro_f05"]
    return res, (keep_e if res["chosen"] == "expected_f" else keep_t)


def cmd_evaluate(args):
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    s1 = ids("train")[0]
    half = half_of(s1["id"].to_list())
    val = sc.fold == 0
    cal, ev = val & (half == 0), val & (half == 1)
    cty = s1["country"].to_numpy()
    base = pl.read_parquet(os.path.join(CACHE, "feats", OOF_GNN))           # a, b, fold, y, p (GNN OOF, chain A on fold 0)
    iso12 = fit_iso(*[base.filter(pl.col("fold").is_in([1, 2])).sample(n=8_000_000, seed=0)[c].to_numpy() for c in ("p", "y")])
    v = (base["fold"] == 0).to_numpy()
    lg = {"A": np.log(np.clip(base["p"].to_numpy()[v], 1e-7, 1 - 1e-7) / np.clip(1 - base["p"].to_numpy()[v], 1e-7, 1))}
    keys = base.filter(pl.Series(v)).select(["a", "b"])
    for k in ("B", "R"):
        fp = os.path.join(CACHE, "gnn", f"chain{k}_train_scores.parquet")
        if os.path.exists(fp):
            s = keys.join(pl.read_parquet(fp), on=["a", "b"], how="left")
            assert s["gnn"].null_count() == 0
            lg[k] = s["gnn"].to_numpy()
    if "B" in lg:
        lg["ens"] = (lg["A"] + lg["B"]) / 2
    y0 = base["y"].to_numpy()[v]
    a0 = base["a"].to_numpy()[v]
    ctx_p = iso12.predict(base["p"].to_numpy()).astype(np.float32)          # other folds: their own calibration
    rep = {"n_val_s1": int(val.sum()), "n_cal": int(cal.sum()), "n_eval": int(ev.sum())}
    for name, L in lg.items():
        with Timer(f"system {name}"):
            r = {}
            # as V3 did it: isotonic from folds 1-2 of the single-chain OOF, rule from decode_eval (expected-F, gamma 1)
            p = ctx_p.copy()
            p[v] = iso12.predict(sig(L)).astype(np.float32)
            D = base.select(["a", "b"]).with_columns(pl.Series("p", p))
            keep, _ = expected_f_decode(assign(D, floor=0.01), gamma=1.0, M=4096)
            r["iso_folds12_gamma1"] = {"eval": sc.metrics(keep, ev)[0], "val_all": sc.metrics(keep, val)[0]}
            # own calibration on the cal half of fold 0, rule chosen on the cal half
            cm = cal[a0]
            iso = fit_iso(sig(L[cm]), y0[cm])
            p = ctx_p.copy()
            p[v] = iso.predict(sig(L)).astype(np.float32)
            D = base.select(["a", "b"]).with_columns(pl.Series("p", p))
            res, keep = choose_rule(D, sc, cal, ev)
            f, _, _ = sc.per_entity(keep)
            res["by_country_eval"] = {c: float(f[ev & (cty == c)].mean()) for c in sorted(set(cty[ev].tolist()))}
            r["own_cal"] = res
            json.dump({"X": iso.X_thresholds_.tolist(), "Y": iso.y_thresholds_.tolist(), "rule": res["chosen"],
                       "settings": {k: v_ for k, v_ in res[res["chosen"]].items() if k in ("thr", "margin", "gamma")}},
                      open(os.path.join(RDIR, f"calib_{name}.json"), "w"))
            rep[name] = r
            print(name, json.dumps({"as_v3": r["iso_folds12_gamma1"]["eval"]["macro_f05"], "own_cal": res["eval_f05"],
                                    "rule": res["chosen"], "by_country": res["by_country_eval"]}), flush=True)
    json.dump(rep, open(os.path.join(RDIR, "evaluate.json"), "w"), indent=1)


def cmd_final(args):
    """test: mean of the chain A / B GNN logits (or one chain) -> calibration and rule of that system -> files."""
    t0 = time.time()
    cj = json.load(open(os.path.join(RDIR, f"calib_{args.system}.json")))
    parts = {"ens": ["A", "B"], "A": ["A"], "B": ["B"]}[args.system]
    T = None
    for k in parts:
        s = pl.read_parquet(os.path.join(CACHE, "gnn", f"chain{k}_test_scores.parquet"))
        T = s.rename({"gnn": k}) if T is None else T.join(s.rename({"gnn": k}), on=["a", "b"], how="left")
    L = np.mean(np.stack([T[k].to_numpy() for k in parts]), 0)
    p = np.interp(sig(L), cj["X"], cj["Y"]).astype(np.float32)
    T = T.select(["a", "b"]).with_columns(pl.Series("p", p))
    st = cj["settings"]
    if cj["rule"] == "threshold":
        keep = decide_records(best_per_record(T), st["thr"], st["margin"])
    else:
        keep, _ = expected_f_decode(assign(T, floor=0.01), gamma=st["gamma"], M=4096)
    odir = os.path.join(RESULTS, "final", args.out, "output")
    os.makedirs(odir, exist_ok=True)
    s1, rr = ids("test")
    s1id, rid = s1["id"].to_numpy(), rr["id"].to_numpy()
    order = s1id.tolist()
    match = keep.group_by("a").agg(pl.col("b"))
    write_id_lists(os.path.join(odir, "matching_results.tsv"), order,
                   {s1id[a]: [rid[x] for x in bs] for a, bs in match.iter_rows()}, "matched_entity_ids")
    if not args.no_candidates:
        write_id_lists(os.path.join(odir, "candidate_pairs.tsv"), order,
                       {s1id[a]: [rid[x] for x in bs] for a, bs in T.group_by("a").agg(pl.col("b")).iter_rows()},
                       "candidate_entity_ids")
    cty = s1["country"].to_numpy()
    has = np.zeros(len(order), bool)
    has[match["a"].to_numpy()] = True
    stats = {"system": args.system, "rule": cj["rule"], "settings": st, "pred_matches": keep.height,
             "s1_empty": int((~has).sum()), "by_country": {}, "runtime_s": time.time() - t0, "gpu": gpu_mem()}
    for c in sorted(set(cty.tolist())):
        m = cty == c
        stats["by_country"][c] = {"s1": int(m.sum()), "with_match": int(has[m].sum()),
                                  "matches": int((cty[keep["a"].to_numpy()] == c).sum())}
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(os.path.join(RESULTS, "final", args.out, "test_stats.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["refit", "p1", "extras", "evaluate", "final"])
    ap.add_argument("--system", default="ens")
    ap.add_argument("--out", default="final_v4")
    ap.add_argument("--no_candidates", action="store_true")
    args = ap.parse_args()
    gpu_init()
    os.makedirs(RDIR, exist_ok=True)
    {"refit": cmd_refit, "p1": cmd_p1, "extras": cmd_extras, "evaluate": cmd_evaluate, "final": cmd_final}[args.step](args)


if __name__ == "__main__":
    main()
