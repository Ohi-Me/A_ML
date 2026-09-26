"""Leave-one-country-out: fit every learned stage on one country (default India), evaluate on the other (US), next to
the in-domain score (India fold 0). Question: do the GNN / XLM-R components generalise to an unseen country, or only
improve in-domain validation? (France on test is unseen.)

Same cross-fitting as the main pipeline, restricted to the source country:
  half A fits source folds {1,2} (XGBoost early stopping on source fold 3) and scores folds {0,3,4} of every country;
  half B fits source folds {3,4} and scores folds {1,2}. Calibration / thresholds use source folds 1-2 only.
The target country's labels are only used for scoring. Blend weights as on test: 0.75 full / 0.25 no-embedding for
the source country, 0.5 / 0.5 for the target country (what V2 / V3 did for France).

Leakage note: the bi-encoder (cos_e3, r_e3_emb) and the candidate retrieval were trained on both countries. --noemb
removes every embedding input (no-embedding stage 1 only, GNN without cos_e3) for a leakage-free run; report both.

Steps (each resumable; outputs in data/cache_v2/p2/loco_<src>2<dst>[_noemb]/):
  python code/phase2/loco_eval.py stage1   [--src India --dst US] [--noemb]
  python code/phase2/loco_eval.py xenc     (XLM-R base fine-tuned on source band pairs; --xenc_max_train 400000)
  python code/phase2/loco_eval.py gnn      (GNN with and without the XLM-R edge input)
  python code/phase2/loco_eval.py pl       (V5: pseudo-label self-training on the unseen country, see code/v5/pl_lib.py)
  python code/phase2/loco_eval.py evaluate -> results/phase2/loco_<src>2<dst>[_noemb][_<name>].json
V5 options: --feats_table c2v5_train_sim19 --full_tag xgb_v5_full --noemb_tag xgb_v5_noemb --name v5 --skip_neural
(stage1 -> pl -> evaluate only). evaluate also searches the expected-F0.5 gamma on the unseen country
('best_gamma_unseen'), which final_v5.py uses for France.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, RESULTS, load_source  # noqa: E402
from common.xgbdata import PartIter  # noqa: E402
from gnn.edge_gnn import FEATS  # noqa: E402
from phase2 import gnn_lib as GL  # noqa: E402
from phase2.p2lib import BAND, decode, device, dump, fit_iso, logit, sig, tune_threshold  # noqa: E402

FEATS_TABLE = "c2_train_sim19"
FULL, NOEMB = "xgb_c2_sim19_v2", "xgb_c2_sim19_noemb_v2"


def tag(args):
    return f"loco_{args.src}2{args.dst}{'_noemb' if args.noemb else ''}{'_' + args.name if args.name else ''}"


def odir(args):
    d = os.path.join(CACHE, "p2", tag(args))
    os.makedirs(d, exist_ok=True)
    return d


def src_mask(args):
    return ids("train")[0]["country"].to_numpy() == args.src


def cmd_stage1(args):
    import xgboost as xgb
    out = os.path.join(odir(args), "stage1.parquet")
    if os.path.exists(out):
        print("exists", out)
        return
    files = sorted(glob.glob(os.path.join(CACHE, "feats", FEATS_TABLE, "part_*.parquet")))
    am = src_mask(args)
    tags = [NOEMB] if args.noemb else [FULL, NOEMB]
    models, rep = {}, {}
    for t in tags:
        rp = json.load(open(os.path.join(RESULTS, "xgboost", t, "report.json")))
        feats, params = rp["features"], dict(rp["params"])
        mb = int(params.get("max_bin", 256))
        with Timer(f"{t}: half A on {args.src} folds 1,2 (early stop {args.src} fold 3)"):
            dtr = xgb.QuantileDMatrix(PartIter(files, feats, [1, 2], a_mask=am), max_bin=mb)
            des = xgb.QuantileDMatrix(PartIter(files, feats, [3], a_mask=am), ref=dtr)
            bA = xgb.train(params, dtr, args.rounds, evals=[(des, "fold3")], early_stopping_rounds=150, verbose_eval=250)
            n = bA.best_iteration + 1
            bA = bA[:n]
            del dtr, des
        with Timer(f"{t}: half B on {args.src} folds 3,4 ({n} rounds)"):
            dtr = xgb.QuantileDMatrix(PartIter(files, feats, [3, 4], a_mask=am), max_bin=mb)
            bB = xgb.train(params, dtr, n)
            del dtr
        models[t] = (feats, bA, bB)
        rep[t] = {"rounds": n}
    cty = ids("train")[0]["country"].to_numpy()
    outs = []
    with Timer("score every pair (A: folds 0,3,4; B: folds 1,2)"):
        for f in files:
            d = pl.read_parquet(f)
            fold = d["fold"].to_numpy()
            ia = np.isin(fold, [0, 3, 4])
            cols = {}
            for t, (feats, bA, bB) in models.items():
                X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
                p = np.zeros(len(fold), np.float32)
                if ia.any():
                    p[ia] = bA.predict(xgb.DMatrix(X[ia]))
                if (~ia).any():
                    p[~ia] = bB.predict(xgb.DMatrix(X[~ia]))
                cols[t] = p
            if args.noemb:
                p = cols[NOEMB]
            else:
                w = np.where(cty[d["a"].to_numpy()] == args.src, 0.75, args.w_unseen).astype(np.float32)
                p = w * cols[FULL] + (1 - w) * cols[NOEMB]
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p.astype(np.float32)),
                                                                      *[pl.Series(f"p_{t}", v) for t, v in cols.items()]))
    pl.concat(outs).write_parquet(out)
    dump(rep, tag(args), "stage1_step.json")


def cmd_xenc(args):
    import torch
    from neural_reranker.ml_cross_encoder import PairModel, score, tokenize, train
    from neural_reranker.prep_pairs import txt
    dev = device()
    out = os.path.join(odir(args), "xenc.parquet")
    if os.path.exists(out):
        print("exists", out)
        return
    S = pl.read_parquet(os.path.join(odir(args), "stage1.parquet"), columns=["a", "b", "fold", "y", "p"])
    H = S.filter((pl.col("p") >= BAND[0]) & (pl.col("p") <= BAND[1]))
    cty = ids("train")[0]["country"].to_numpy()
    s1 = load_source("train", 1)
    rr = pl.concat([load_source("train", 2), load_source("train", 3)])
    a, b = H["a"].to_numpy(), H["b"].to_numpy()
    t1 = txt(s1["business_name"].to_numpy()[a], s1["business_address"].to_numpy()[a], 110, True)
    t2 = txt(rr["business_name"].to_numpy()[b], rr["business_address"].to_numpy()[b], 110, True)
    del s1, rr
    with Timer(f"tokenize {H.height} band pairs"):
        tok, pad = tokenize(t1, t2)
    fold, y = H["fold"].to_numpy(), H["y"].to_numpy()
    src = cty[a] == args.src
    x = np.zeros(H.height, np.float32)
    for name, fit_f, score_f in GL.HALVES:
        tr = np.where(src & np.isin(fold, fit_f))[0]
        if len(tr) > args.xenc_max_train:
            tr = np.random.default_rng(0).choice(tr, args.xenc_max_train, replace=False)
        with Timer(f"XLM-R {name}: fine-tune on {len(tr)} {args.src} band pairs"):
            m = train(tok[tr], pad, y[tr], dev, args.xenc_epochs, seed=ord(name))
        te = np.where(np.isin(fold, score_f))[0]
        with Timer(f"XLM-R {name}: score {len(te)} pairs"):
            x[te] = score(m, tok[te], pad, dev)
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    H.select(["a", "b"]).with_columns(pl.Series("mlxenc", x)).write_parquet(out)


def cmd_gnn(args):
    dev = device()
    d = odir(args)
    feats = [f for f in FEATS if not (args.noemb and f == "cos_e3")]
    am = src_mask(args)
    S = pl.read_parquet(os.path.join(d, "stage1.parquet"), columns=["a", "b", "p"])
    for name, extras in (("gnn", []), ("gnn_xenc", [(os.path.join(d, "xenc.parquet"), "mlxenc")])):
        out = os.path.join(d, f"{name}.parquet")
        if os.path.exists(out) or (extras and not os.path.exists(extras[0][0])):
            print("skip", name)
            continue
        with Timer(f"LOCO {name}"):
            G = GL.build_graph(FEATS_TABLE, S, dev, extras, feats)
            lg = GL.crossfit(G, os.path.join(d, name), am, args.epochs, args.batch_s1)
            G.M.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", sig(lg).astype(np.float32))).write_parquet(out)
            del G


def cmd_pl(args):
    """V5 self-training check: pseudo-label the unseen country with the source-trained stage 1, refit stage 1 on
    source folds 1-4 with and without the pseudo-labelled rows, score the unseen country with both."""
    import xgboost as xgb
    from v5.pl_lib import AGREE_COLS, MixIter, select_pseudo
    d = odir(args)
    if os.path.exists(os.path.join(d, "stage1_pl.parquet")):
        print("exists", d, "stage1_pl")
        return
    files = sorted(glob.glob(os.path.join(CACHE, "feats", FEATS_TABLE, "part_*.parquet")))
    cty = ids("train")[0]["country"].to_numpy()
    S = pl.read_parquet(os.path.join(d, "stage1.parquet"), columns=["a", "b", "fold", "y", "p"])
    srcr = S.filter(pl.Series(cty[S["a"].to_numpy()] == args.src) & pl.col("fold").is_in([1, 2]))
    srcr = srcr.sample(n=min(8_000_000, srcr.height), seed=0)
    iso = fit_iso(srcr["p"].to_numpy(), srcr["y"].to_numpy())
    dst = S.filter(pl.Series(cty[S["a"].to_numpy()] == args.dst)).select(["a", "b", "p"])
    dst = dst.with_columns(pl.Series("p", iso.predict(dst["p"].to_numpy()).astype(np.float32)))
    F = pl.concat([pl.read_parquet(f, columns=["a", "b"] + AGREE_COLS) for f in files]).join(dst.select(["a", "b"]), on=["a", "b"], how="semi")
    rows, rep = select_pseudo(dst, F, args.pos_thr, args.margin, args.neg_thr, args.addr_jac, args.max_pl_pairs)
    true = S.select(["a", "b", "y"]).rename({"y": "y_true"})
    chk = rows.join(true, on=["a", "b"], how="left")
    pos = chk.filter(pl.col("y") == 1)
    rep["pseudo_label_accuracy"] = float((chk["y"] == chk["y_true"]).mean()) if chk.height else None
    rep["pseudo_positive_precision"] = float(pos["y_true"].mean()) if pos.height else None
    print("pseudo-labels", json.dumps(rep), flush=True)
    rounds = json.load(open(os.path.join(RESULTS, "phase2", tag(args), "stage1_step.json")))
    am = src_mask(args)
    tags = [NOEMB] if args.noemb else [FULL, NOEMB]
    preds = {"all": {}, "pl": {}}
    for t in tags:
        rp = json.load(open(os.path.join(RESULTS, "xgboost", t, "report.json")))
        feats, params = rp["features"], dict(rp["params"])
        n = int(rounds[t]["rounds"] * 1.25)
        for kind in ("all", "pl"):
            with Timer(f"{t} {kind}: source folds 1-4{' + pseudo-labels' if kind == 'pl' else ''}, {n} rounds"):
                it = MixIter(files, feats, [1, 2, 3, 4], a_mask=am, extra_files=files if kind == "pl" else (),
                             pl_rows=rows if kind == "pl" else None)
                m = xgb.train(params, xgb.QuantileDMatrix(it, max_bin=int(params.get("max_bin", 256))), n)
            outs = []
            for f in files:
                q = pl.read_parquet(f, columns=["a", "b"] + feats)
                q = q.filter(pl.Series(cty[q["a"].to_numpy()] == args.dst))
                outs.append(q.select(["a", "b"]).with_columns(pl.Series(
                    "p", m.predict(xgb.DMatrix(q.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy())))))
            preds[kind][t] = pl.concat(outs)
    for kind in ("all", "pl"):
        if args.noemb:
            P = preds[kind][NOEMB]
        else:
            P = preds[kind][FULL].join(preds[kind][NOEMB].rename({"p": "p2"}), on=["a", "b"]).with_columns(
                (args.w_unseen * pl.col("p") + (1 - args.w_unseen) * pl.col("p2")).alias("p")).drop("p2")
        out = S.join(P.rename({"p": "p_new"}), on=["a", "b"], how="left").with_columns(
            pl.coalesce(["p_new", "p"]).cast(pl.Float32).alias("p")).drop("p_new")
        out.write_parquet(os.path.join(d, f"stage1_{kind}.parquet"))
    dump(rep, tag(args), "pl_step.json")


def ece(p, y, bins=15):
    e, n = 0.0, len(p)
    if n == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def cmd_evaluate(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from common.decode import assign
    dev = device()
    d = odir(args)
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    cty = sc.country
    masks = {f"{args.src}_f0_in_domain": (sc.fold == 0) & (cty == args.src),
             f"{args.dst}_f0_unseen": (sc.fold == 0) & (cty == args.dst),
             f"{args.dst}_all_unseen": (sc.fold >= 0) & (cty == args.dst)}
    S = pl.read_parquet(os.path.join(d, "stage1.parquet"), columns=["a", "b", "fold", "y", "p"])
    systems = {"stage1": S}
    if os.path.exists(os.path.join(d, "xenc.parquet")):
        X = S.join(pl.read_parquet(os.path.join(d, "xenc.parquet")), on=["a", "b"], how="left")
        x = X["mlxenc"].to_numpy().astype(np.float64)
        h = ~np.isnan(x)
        x0 = np.where(h, x, 0)
        lp = logit(X["p"].to_numpy())
        Z = np.column_stack([lp, x0, h, lp * x0])
        srcm = cty[X["a"].to_numpy()] == args.src
        tr = h & srcm & np.isin(X["fold"].to_numpy(), [1, 2])
        lr = LogisticRegression(C=1.0, max_iter=300).fit(Z[tr], X["y"].to_numpy()[tr])
        p = X["p"].to_numpy().astype(np.float64)
        p[h] = lr.predict_proba(Z[h])[:, 1]
        systems["stage1+xlmr_stack"] = X.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p.astype(np.float32)))
    for name in ("stage1_all", "stage1_pl"):
        if os.path.exists(os.path.join(d, f"{name}.parquet")):
            systems[name] = pl.read_parquet(os.path.join(d, f"{name}.parquet"), columns=["a", "b", "fold", "y", "p"])
    for name in ("gnn", "gnn_xenc"):
        if os.path.exists(os.path.join(d, f"{name}.parquet")):
            systems[{"gnn": "gnn_no_xlmr", "gnn_xenc": "gnn+xlmr"}[name]] = pl.read_parquet(os.path.join(d, f"{name}.parquet"))
    rep = {"args": vars(args), "masks": {k: int(v.sum()) for k, v in masks.items()}, "systems": {}}
    for name, D in systems.items():
        with Timer(f"evaluate {name}"):
            srcp = D.filter(pl.Series(cty[D["a"].to_numpy()] == args.src))
            cal = srcp.filter(pl.col("fold").is_in([1, 2]))
            cal = cal.sample(n=min(8_000_000, cal.height), seed=0)
            iso = fit_iso(cal["p"].to_numpy(), cal["y"].to_numpy())
            sub = Scorer.__new__(Scorer)
            sub.__dict__ = dict(sc.__dict__)
            sub.fold = np.where(cty == args.src, sc.fold, -1).astype(np.int8)   # thresholds tuned on source folds 1-2
            tt = tune_threshold(D.select(["a", "b", "p"]), sub)
            res = {"threshold": tt}
            for dec_name, kw in (("iso+ef", dict(rule="ef", iso=iso)), ("raw+ef", dict(rule="ef", iso=None)),
                                 ("thr+margin", dict(rule="thr", thr=tt["thr"], margin=tt["margin"]))):
                keep, _ = decode(D.select(["a", "b", "p"]), dev=dev, M=2048, **kw)
                r = {}
                for mn, m in masks.items():
                    mm, _ = sc.metrics(keep, m)
                    mm["pred_over_true"] = mm["n_pred_matches"] / max(int(sc.n_true[m].sum()), 1)
                    r[mn] = {k: mm[k] for k in ("macro_f05", "pair_precision", "pair_recall", "fp", "fn", "n_pred_matches",
                                                "pred_over_true", "n_pred_empty", "singleton_acc")}
                res[dec_name] = r
            # decoder strictness for an unseen country: expected-F0.5 gamma that is best on the target country
            gg = {}
            for g in args.gammas:
                keep, _ = decode(D.select(["a", "b", "p"]), rule="ef", iso=iso, gamma=g, dev=dev, M=2048)
                gg[str(g)] = {mn: sc.metrics(keep, m)[0]["macro_f05"] for mn, m in masks.items()}
            res["gamma_grid"] = gg
            um = f"{args.dst}_all_unseen"
            res["best_gamma_unseen"] = float(max(args.gammas, key=lambda g: gg[str(g)][um]))
            # ranking quality and calibration on each population
            for mn, m in masks.items():
                mD = D.filter(pl.Series(m[D["a"].to_numpy()]))
                yy, pp = mD["y"].to_numpy(), mD["p"].to_numpy()
                A = assign(mD.select(["a", "b", "p"]).with_columns(pl.Series("p", iso.predict(pp).astype(np.float32))), floor=0.01)
                ya = (sc.true_s1_of_b[A["b"].to_numpy()] == A["a"].to_numpy()).astype(float)
                res.setdefault("scores", {})[mn] = {
                    "auc": float(roc_auc_score(yy, pp)) if 0 < yy.sum() < len(yy) else None,
                    "ap": float(average_precision_score(yy, pp)) if yy.sum() else None,
                    "ece_assigned_calibrated": ece(A["p"].to_numpy(), ya),
                    "mean_cal_p_assigned": float(A["p"].mean()), "true_rate_assigned": float(ya.mean())}
            rep["systems"][name] = res
            print(name, json.dumps({k: v[f"{args.dst}_f0_unseen"]["macro_f05"] for k, v in res.items() if k in ("iso+ef", "raw+ef", "thr+margin")}), flush=True)
    # component deltas: in-domain vs unseen
    s = rep["systems"]
    deltas = {}
    for a_, b_ in (("stage1+xlmr_stack", "stage1"), ("gnn_no_xlmr", "stage1"), ("gnn+xlmr", "gnn_no_xlmr"), ("gnn+xlmr", "stage1"),
                   ("stage1_pl", "stage1_all")):
        if a_ in s and b_ in s:
            deltas[f"{a_} - {b_}"] = {mn: s[a_]["iso+ef"][mn]["macro_f05"] - s[b_]["iso+ef"][mn]["macro_f05"] for mn in masks}
    rep["component_deltas_iso_ef"] = deltas
    dump(rep, f"{tag(args)}.json")


def main():
    global FEATS_TABLE, FULL, NOEMB
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["stage1", "xenc", "gnn", "pl", "evaluate", "all"])
    ap.add_argument("--src", default="India")
    ap.add_argument("--dst", default="US")
    ap.add_argument("--noemb", action="store_true")
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--xenc_max_train", type=int, default=400_000)
    ap.add_argument("--xenc_epochs", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_s1", type=int, default=1500)
    ap.add_argument("--feats_table", default=FEATS_TABLE)
    ap.add_argument("--full_tag", default=FULL)
    ap.add_argument("--noemb_tag", default=NOEMB)
    ap.add_argument("--name", default="", help="suffix of the output names (e.g. v5)")
    ap.add_argument("--w_unseen", type=float, default=0.5)
    ap.add_argument("--skip_neural", action="store_true", help="'all' runs stage1 -> pl -> evaluate (no XLM-R / GNN)")
    ap.add_argument("--with_pl", action="store_true", help="'all' also runs the pseudo-label step")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--pos_thr", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--neg_thr", type=float, default=0.02)
    ap.add_argument("--addr_jac", type=float, default=0.3)
    ap.add_argument("--max_pl_pairs", type=int, default=4_000_000)
    args = ap.parse_args()
    FEATS_TABLE, FULL, NOEMB = args.feats_table, args.full_tag, args.noemb_tag
    t0 = time.time()
    if args.step == "all":
        steps = ["stage1"] + ([] if args.skip_neural else ["xenc", "gnn"]) + (["pl"] if args.with_pl or args.skip_neural else []) + ["evaluate"]
    else:
        steps = [args.step]
    for s in steps:
        {"stage1": cmd_stage1, "xenc": cmd_xenc, "gnn": cmd_gnn, "pl": cmd_pl, "evaluate": cmd_evaluate}[s](args)
    print("done in", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
