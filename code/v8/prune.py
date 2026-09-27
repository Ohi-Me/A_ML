"""V8 candidate pruner: a wide candidate union for recall, cut back to a few pairs per record before the heavy stages.

Why: the V7 union raises candidate recall but also the pair count (budget 12 per record, C2 had 8.2), and the
string-feature stage keeps the whole table in memory (V7 needs >= 128 GB). A light model on features that cost
almost nothing (retrieval ranks, pass flags, TF-IDF and bi-encoder cosines, their competition among the record's
and the S1's candidates, exact-equality flags, name-key frequencies) removes the pairs that are clearly not a match.
Everything downstream (pair features, stage 1, cross-encoders, collective rounds) then runs on the pruned table,
which is also the final candidate_pairs.tsv.

Steps
  feats  light features for a candidate table (train: with fold and label)      -> feats/<cands>_<split>_lite/
  fit    XGBoost pruner in the usual halves (A: folds 1,2, early stop on fold 3, scores 0,3,4; B: folds 3,4 -> 1,2)
         and the cut, fixed before any downstream number: keep a pair if its pruner score >= tau, or if it is among
         the record's top 2 (keeps the runner-up for the competition features); at most --max_per_record per record.
         tau = the --max_loss quantile of the true pairs' scores on the tuning folds 1-2 (default: lose <= 0.05 % of
         the true pairs the union found). Reported on fold 0.
  apply  write the pruned table cands/<out>_<split>.parquet (same columns as the input: a, b, r_*, x_*). Train uses
         the out-of-fold score, test the mean of the two halves.

  python code/v8/prune.py feats --cands c3wdms --split train --universe dms
  python code/v8/prune.py feats --cands c3w --split test
  python code/v8/prune.py fit   --cands c3wdms
  python code/v8/prune.py apply --cands c3wdms --split train --out c4dms
  python code/v8/prune.py apply --cands c3w --split test --out c4 --fit_from c3wdms
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, NORM, RESULTS, load_gt  # noqa: E402
from common.split import fold_of  # noqa: E402
from common.xgbdata import PartIter  # noqa: E402

RDIR = os.path.join(RESULTS, "v8")
NON_FEATURES = {"a", "b", "fold", "y"}


def _h(s):
    """u64 hash per row, 0 for empty (so two empty fields never count as equal)."""
    s = pl.Series(s).fill_null("")
    h = s.hash(seed=0).to_numpy().astype(np.uint64)
    h[(s == "").to_numpy()] = 0
    return h


def rowdot(M, a, b, off):
    return np.asarray(M[a].multiply(M[b + off]).sum(axis=1)).ravel().astype(np.float32)


def cmd_feats(args):
    """Built in record-aligned chunks so the wide table never sits in memory as a whole: global per-pair columns are
    numpy arrays, the per-S1 competition is computed once on a slim frame, everything per record inside the chunk."""
    t0 = time.time()
    C = pl.read_parquet(os.path.join(CACHE, "cands", f"{args.cands}_{args.split}.parquet"))
    assert C.select((pl.col("b").diff().fill_null(0) >= 0).all()).item(), "candidate table must be sorted by b"
    # the test lists come from the all-folds bi-encoder (r_e2f_emb); the pruner knows them by the train name
    C = C.rename({"r_e2f_emb": "r_e3_emb"}, strict=False)
    keep_cols = [c for c in C.columns if c.startswith("r_") or c.startswith("x_")]
    cols = ["id", "country", "key", "n", "am", "first_num", "state"]

    def rd(k):
        path = os.path.join(CACHE, f"norm_{NORM}_{args.split}_s{k}.parquet")
        have = pl.read_parquet_schema(path)
        d = pl.read_parquet(path, columns=[c for c in cols if c in have])
        return d if "n" in d.columns else d.with_columns(pl.col("key").alias("n"))
    s1 = rd(1)
    rr = pl.concat([rd(2), rd(3)])
    n1, npairs = s1.height, C.height
    a = C["a"].to_numpy().astype(np.int64)
    b = C["b"].to_numpy().astype(np.int64)
    print("pairs", npairs, "records", C["b"].n_unique(), flush=True)
    with Timer("tf-idf cosines"):
        bdir = os.path.join(CACHE, "blocking", f"b1_{args.split}")
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        cn = np.zeros(npairs, np.float32)
        ca = np.zeros(npairs, np.float32)
        for s in range(0, npairs, 2_000_000):
            e = min(s + 2_000_000, npairs)
            cn[s:e] = rowdot(NAME, a[s:e], b[s:e], n1)
            ca[s:e] = rowdot(ADDR, a[s:e], b[s:e], n1)
        del NAME, ADDR
    with Timer("bi-encoder pair cosine"):
        pc = pl.read_parquet(os.path.join(CACHE, "emb", f"e3_{args.split}", f"paircos_{args.cands}.parquet"))
        if pc.height == npairs and (pc["b"].to_numpy() == b).all() and (pc["a"].to_numpy() == a).all():
            ce = pc["cos_e3"].to_numpy().astype(np.float32)
        else:
            ce = C.select(["a", "b"]).join(pc, on=["a", "b"], how="left", maintain_order="left")["cos_e3"]
            ce = ce.fill_null(0.0).to_numpy().astype(np.float32)
        del pc
    cs = cn + ca
    with Timer("per-S1 competition (slim frame)"):
        S = pl.DataFrame({"a": a, "cs": cs, "ce": ce})
        S = S.with_columns(pl.len().over("a").cast(pl.Int32).alias("a_n"),
                           (pl.col("cs") - pl.col("cs").max().over("a")).alias("a_gap_cos_sum"),
                           pl.col("cs").rank("ordinal", descending=True).over("a").cast(pl.Int32).alias("a_rank_cos_sum"),
                           (pl.col("ce") - pl.col("ce").max().over("a")).alias("a_gap_cos_e3"),
                           pl.col("ce").rank("ordinal", descending=True).over("a").cast(pl.Int32).alias("a_rank_cos_e3"))
        A_SIDE = {c: S[c].to_numpy() for c in ("a_n", "a_gap_cos_sum", "a_rank_cos_sum", "a_gap_cos_e3", "a_rank_cos_e3")}
        del S
    with Timer("hashes and key frequencies"):
        H = {nm: (_h(s1[c]), _h(rr[c])) for c, nm in (("key", "key_eq"), ("n", "name_eq"), ("am", "addr_eq"),
                                                         ("first_num", "fnum_eq"), ("state", "state_eq"))}
        am1, amr = (s1["am"].fill_null("") == "").to_numpy(), (rr["am"].fill_null("") == "").to_numpy()
        # name-key frequency among the S1 of the universe (train) / of the test index, and among its records
        present = np.zeros(n1, bool)
        present[np.unique(a)] = True
        kept = np.ones(rr.height, bool)
        g = None
        if args.split == "train":
            pairs, _ = load_gt()
            g = pairs.join(pl.DataFrame({"s1": s1["id"], "a": np.arange(n1, dtype=np.int64)}), on="s1")
            g = g.join(pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height, dtype=np.int64)}), on="rid").select(["a", "b"])
            if args.universe != "none":
                from v7.miss_probe import universe
                true_a = np.full(rr.height, -1, np.int64)
                true_a[g["b"].to_numpy()] = g["a"].to_numpy()
                present, kept = universe(args.universe, s1, rr, true_a)
            g = g.with_columns(pl.lit(1, pl.Int8).alias("y"))
        k1 = s1.filter(pl.Series(present)).group_by(["country", "key"]).len().rename({"len": "key_n_s1"})
        kr = rr.filter(pl.Series(kept)).group_by(["country", "key"]).len().rename({"len": "key_n_r"})
        f1 = s1.select(["country", "key"]).join(k1, on=["country", "key"], how="left", maintain_order="left")
        f1 = f1["key_n_s1"].fill_null(0).to_numpy().astype(np.int32)
        fr = rr.select(["country", "key"]).join(kr, on=["country", "key"], how="left", maintain_order="left")
        fr = fr["key_n_r"].fill_null(0).to_numpy().astype(np.int32)
        fold_arr = np.array([fold_of(x) for x in s1["id"].to_list()], np.int8)
        del s1, rr
    out = os.path.join(CACHE, "feats", f"{args.cands}_{args.split}_lite")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    pos, i, cols_out = 0, 0, []
    with Timer("record-aligned chunks"):
        while pos < npairs:
            e = min(pos + args.part, npairs)
            if e < npairs:
                e = int(np.searchsorted(b, b[e], side="left"))        # never split a record
                if e <= pos:
                    e = int(np.searchsorted(b, b[pos], side="right"))
            sl = slice(pos, e)
            aa, bb = a[sl], b[sl]
            X = C[pos:e].select(["a", "b"] + keep_cols).with_columns(
                pl.Series("cos_name", cn[sl]), pl.Series("cos_addr", ca[sl]), pl.Series("cos_sum", cs[sl]),
                pl.Series("cos_e3", ce[sl]),
                *[pl.Series(nm, ((h1[aa] != 0) & (h1[aa] == hr[bb])).astype(np.int8)) for nm, (h1, hr) in H.items()],
                pl.Series("rec_addr_empty", amr[bb].astype(np.int8)), pl.Series("s1_addr_empty", am1[aa].astype(np.int8)),
                pl.Series("key_n_s1", f1[aa]), pl.Series("key_n_r", fr[bb]), pl.Series("fold", fold_arr[aa]),
                *[pl.Series(k, v[sl]) for k, v in A_SIDE.items()])
            X = X.with_columns(pl.len().over("b").cast(pl.Int16).alias("b_n"))
            for col in ("cos_sum", "cos_e3"):
                X = X.with_columns((pl.col(col) - pl.col(col).max().over("b")).alias(f"b_gap_{col}"),
                                   pl.col(col).rank("ordinal", descending=True).over("b").cast(pl.Int16).alias(f"b_rank_{col}"))
                sec = X.filter(pl.col(f"b_rank_{col}") == 2).select(["b", pl.col(col).alias("_sec")])
                X = X.join(sec, on="b", how="left", maintain_order="left").with_columns(
                    (pl.col(col) - pl.col("_sec").fill_null(0.0)).alias(f"b_gap2_{col}")).drop("_sec")
            if g is not None:
                gi = g.filter((pl.col("b") >= int(bb[0])) & (pl.col("b") <= int(bb[-1]))).with_columns(
                    pl.col("a").cast(X.schema["a"]), pl.col("b").cast(X.schema["b"]))
                X = X.join(gi, on=["a", "b"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
            assert (X["b"].to_numpy() == bb).all() and (X["a"].to_numpy() == aa).all()
            X.write_parquet(os.path.join(out, f"part_{i:03d}.parquet"))
            cols_out = X.columns
            pos, i = e, i + 1
    rep = {"cands": args.cands, "split": args.split, "pairs": npairs, "parts": i, "cols": cols_out, "runtime_s": time.time() - t0}
    os.makedirs(RDIR, exist_ok=True)
    json.dump(rep, open(os.path.join(RDIR, f"prune_feats_{args.cands}_{args.split}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "cols"}), flush=True)


def lite_files(cands, split):
    return sorted(glob.glob(os.path.join(CACHE, "feats", f"{cands}_{split}_lite", "part_*.parquet")))


def keep_mask(S, tau, top, cap):
    """S: frame b, p (one row per pair, any order). Returns the bool keep mask in S's order."""
    r = S.select(pl.col("p").rank("ordinal", descending=True).over("b").alias("r"))["r"].to_numpy()
    p = S["p"].to_numpy()
    return ((p >= tau) | (r <= top)) & (r <= cap)


def cmd_fit(args):
    t0 = time.time()
    files = lite_files(args.cands, "train")
    cols = pl.read_parquet_schema(files[0])
    feats = [c for c in cols if c not in NON_FEATURES]
    params = {"objective": "binary:logistic", "eval_metric": ["logloss", "aucpr"], "tree_method": "hist", "device": "cuda",
              "max_depth": 8, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "lambda": 2.0,
              "max_bin": 256}
    models = {}
    with Timer("pruner A: folds 1,2 (early stop on fold 3)"):
        dtr = xgb.QuantileDMatrix(PartIter(files, feats, [1, 2]), max_bin=256)
        des = xgb.QuantileDMatrix(PartIter(files, feats, [3]), ref=dtr)
        bst = xgb.train(params, dtr, args.rounds, evals=[(des, "fold3")], early_stopping_rounds=50, verbose_eval=100)
        n_rounds = bst.best_iteration + 1
        models["A"] = bst[:n_rounds]
        del dtr, des
    with Timer("pruner B: folds 3,4"):
        dtr = xgb.QuantileDMatrix(PartIter(files, feats, [3, 4]), max_bin=256)
        models["B"] = xgb.train(params, dtr, n_rounds)
        del dtr
    os.makedirs(RDIR, exist_ok=True)
    for k, m in models.items():
        m.save_model(os.path.join(RDIR, f"pruner_{args.cands}_{k}.json"))
    with Timer("out-of-fold scores"):
        outs = []
        for f in files:
            d = pl.read_parquet(f, columns=["a", "b", "fold", "y"] + feats)
            fold = d["fold"].to_numpy()
            X = d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()
            p = np.zeros(d.height, np.float32)
            ia = np.isin(fold, [0, 3, 4])
            if ia.any():
                p[ia] = models["A"].predict(xgb.DMatrix(X[ia]))
            if (~ia).any():
                p[~ia] = models["B"].predict(xgb.DMatrix(X[~ia]))
            outs.append(d.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p)))
        D = pl.concat(outs)
        D.write_parquet(os.path.join(CACHE, "feats", f"prune_oof_{args.cands}.parquet"))
    with Timer("cut"):
        tune = D["fold"].is_in([1, 2]).to_numpy()
        pos_tune = np.sort(D["p"].to_numpy()[tune & (D["y"].to_numpy() == 1)])
        tau = float(pos_tune[int(np.floor(args.max_loss * len(pos_tune)))]) if len(pos_tune) else 0.0
        keep = keep_mask(D.select(["b", "p"]), tau, args.top, args.max_per_record)
        y = D["y"].to_numpy() == 1
        fold = D["fold"].to_numpy()
        nrec = D["b"].n_unique()
        rep = {"cands": args.cands, "features": feats, "rounds": n_rounds, "params": params, "tau": tau, "top": args.top,
               "max_per_record": args.max_per_record, "max_loss": args.max_loss,
               "pairs_before": int(D.height), "pairs_after": int(keep.sum()),
               "pairs_per_record_before": float(D.height / nrec), "pairs_per_record_after": float(keep.sum() / nrec),
               "by_fold": {}}
        for fo in range(5):
            m = fold == fo
            rep["by_fold"][str(fo)] = {"true_pairs_in_union": int((y & m).sum()),
                                       "kept_share_of_true": float((y & m & keep).sum() / max((y & m).sum(), 1))}
        imp = models["A"].get_score(importance_type="gain")
        rep["importance_gain"] = sorted(((feats[int(k[1:])] if k[1:].isdigit() else k, v) for k, v in imp.items()),
                                        key=lambda t: -t[1])[:30]
        rep["runtime_s"] = time.time() - t0
    json.dump(rep, open(os.path.join(RDIR, f"prune_{args.cands}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("features", "params", "importance_gain")}, indent=1), flush=True)


def cmd_apply(args):
    t0 = time.time()
    src = args.fit_from or args.cands
    rep = json.load(open(os.path.join(RDIR, f"prune_{src}.json")))
    if args.split == "train":
        S = pl.read_parquet(os.path.join(CACHE, "feats", f"prune_oof_{src}.parquet"), columns=["a", "b", "p"])
    else:
        feats = rep["features"]
        models = [xgb.Booster(model_file=os.path.join(RDIR, f"pruner_{src}_{k}.json")) for k in ("A", "B")]
        outs = []
        for f in lite_files(args.cands, args.split):
            d = pl.read_parquet(f, columns=["a", "b"] + feats)
            X = xgb.DMatrix(d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy())
            p = (models[0].predict(X) + models[1].predict(X)) / 2
            outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", p.astype(np.float32))))
        S = pl.concat(outs)
    keep = keep_mask(S.select(["b", "p"]), rep["tau"], rep["top"], rep["max_per_record"])
    K = S.filter(pl.Series(keep)).select(["a", "b"])
    C = pl.read_parquet(os.path.join(CACHE, "cands", f"{args.cands}_{args.split}.parquet"))
    out = C.join(K, on=["a", "b"], how="semi").sort(["b", "a"])
    out.write_parquet(os.path.join(CACHE, "cands", f"{args.out}_{args.split}.parquet"))
    res = {"cands": args.cands, "split": args.split, "out": args.out, "tau": rep["tau"], "pairs_before": C.height,
           "pairs_after": out.height, "records": int(C["b"].n_unique()),
           "pairs_per_record_after": out.height / max(int(C["b"].n_unique()), 1), "runtime_s": time.time() - t0}
    if args.split == "train":
        Y = pl.read_parquet(os.path.join(CACHE, "feats", f"prune_oof_{src}.parquet"), columns=["a", "b", "y", "fold"])
        Y = Y.with_columns(pl.Series("keep", keep))
        res["kept_share_of_true_fold0"] = float(Y.filter((pl.col("y") == 1) & (pl.col("fold") == 0))["keep"].mean())
    json.dump(res, open(os.path.join(RDIR, f"prune_apply_{args.out}_{args.split}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["feats", "fit", "apply"])
    ap.add_argument("--cands", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--universe", default="none", choices=["none", "dms", "sim19"])
    ap.add_argument("--out", default="")
    ap.add_argument("--fit_from", default="", help="apply: pruner fitted on this train candidate table")
    ap.add_argument("--rounds", type=int, default=800)
    ap.add_argument("--max_loss", type=float, default=0.0005, help="share of the union's true pairs the cut may lose (folds 1-2)")
    ap.add_argument("--top", type=int, default=2, help="always keep the record's top-N by pruner score")
    ap.add_argument("--max_per_record", type=int, default=10)
    ap.add_argument("--part", type=int, default=4_000_000)
    args = ap.parse_args()
    gpu_init()
    {"feats": cmd_feats, "fit": cmd_fit, "apply": cmd_apply}[args.step](args)


if __name__ == "__main__":
    main()
