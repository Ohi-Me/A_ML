"""Turn pair scores into matches and score them with the official macro F0.5 (vectorized).

Decision: every S2/S3 record goes to its best-scoring S1 (the ground truth never links a record to two S1),
and only if that score >= thr and beats the record's runner-up S1 by >= margin."""
import json
import os

import numpy as np
import polars as pl

from common.io import CACHE, load_gt, NORM
from common.split import N_FOLDS, VAL_FOLD, fold_of

_ID_CACHE = {}


def ids(split, norm=None):
    norm = norm or NORM
    if split not in _ID_CACHE:
        s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{norm}_{split}_s1.parquet"), columns=["id", "country"])
        rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{norm}_{split}_s{k}.parquet"), columns=["id"])
                        for k in (2, 3)])
        _ID_CACHE[split] = (s1, rr)
    return _ID_CACHE[split]


def best_per_record(D):
    """D: a, b, p -> one row per record: b, a (best), p1, p2 (runner-up score or 0)."""
    s = D.select(["a", "b", "p"]).sort(["b", "p"], descending=[False, True])
    g = s.group_by("b", maintain_order=True).agg(pl.col("a").first(), pl.col("p").first().alias("p1"),
                                                  pl.col("p").slice(1, 1).first().alias("p2"))
    return g.with_columns(pl.col("p2").fill_null(0.0))


def decide_records(best, thr, margin=0.0):
    return best.filter((pl.col("p1") >= thr) & ((pl.col("p1") - pl.col("p2")) >= margin)).select(["a", "b"])


class Scorer:
    """Vectorized macro F0.5 for the training split. Truth as arrays indexed by S1 row."""

    def __init__(self, split="train", drop=None):
        """drop: optional bool array over S1 rows (entities removed by simulate_drop.py); they leave the evaluation
        and their records count as distractors."""
        s1, rr = ids(split)
        self.n1 = s1.height
        self.country = s1["country"].to_numpy()
        self.fold = np.array([fold_of(x) for x in s1["id"].to_list()], np.int8)
        if drop is not None:
            self.fold = np.where(drop, -1, self.fold).astype(np.int8)
        pairs, _ = load_gt()
        m1 = pl.DataFrame({"s1": s1["id"], "a": np.arange(self.n1, dtype=np.int32)})
        mr = pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height, dtype=np.int32)})
        self.T = pairs.join(m1, on="s1").join(mr, on="rid").select(["a", "b"])
        self.n_true = np.bincount(self.T["a"].to_numpy(), minlength=self.n1)
        self.true_s1_of_b = np.full(rr.height, -1, np.int64)
        self.true_s1_of_b[self.T["b"].to_numpy()] = self.T["a"].to_numpy()
        if drop is not None:
            gone = self.true_s1_of_b >= 0
            gone[gone] = drop[self.true_s1_of_b[gone]]
            self.true_s1_of_b[gone] = -1
            self.T = self.T.filter(pl.Series(~drop[self.T["a"].to_numpy()]))
            self.n_true[drop] = 0

    def per_entity(self, pred):
        """pred: frame a, b. Returns F0.5 per S1 row (array n1) + counts."""
        a = pred["a"].to_numpy()
        b = pred["b"].to_numpy()
        n_pred = np.bincount(a, minlength=self.n1)
        tp = np.bincount(a[self.true_s1_of_b[b] == a], minlength=self.n1)
        P = np.divide(tp, n_pred, out=np.zeros(self.n1), where=n_pred > 0)
        R = np.divide(tp, self.n_true, out=np.zeros(self.n1), where=self.n_true > 0)
        den = 0.25 * P + R
        f = np.divide(1.25 * P * R, den, out=np.zeros(self.n1), where=den > 0)
        f[self.n_true == 0] = (n_pred[self.n_true == 0] == 0).astype(float)
        return f, n_pred, tp

    def metrics(self, pred, mask):
        f, n_pred, tp = self.per_entity(pred)
        nt = self.n_true[mask]
        out = {"n_s1": int(mask.sum()), "macro_f05": float(f[mask].mean()),
               "tp": int(tp[mask].sum()), "fp": int((n_pred - tp)[mask].sum()), "fn": int((nt - tp[mask]).sum()),
               "n_pred_matches": int(n_pred[mask].sum()), "n_pred_empty": int((n_pred[mask] == 0).sum()),
               "n_true_singletons": int((nt == 0).sum()),
               "singleton_acc": float(((n_pred[mask] == 0) & (nt == 0)).sum() / max((nt == 0).sum(), 1))}
        out["pair_precision"] = out["tp"] / max(out["tp"] + out["fp"], 1)
        out["pair_recall"] = out["tp"] / max(out["tp"] + out["fn"], 1)
        return out, f

    def breakdown(self, f, mask):
        rows = []
        nt = self.n_true
        buckets = {"singleton": nt == 0, "1 match": nt == 1, "2-3": (nt >= 2) & (nt <= 3), "4-6": (nt >= 4) & (nt <= 6),
                   "7+": nt >= 7}
        for c in sorted(set(self.country.tolist())):
            for bn, bm in buckets.items():
                m = mask & (self.country == c) & bm
                if m.sum():
                    rows.append({"country": c, "bucket": bn, "n": int(m.sum()), "f05": float(f[m].mean()),
                                 "loss_share": float((1 - f[m]).sum() / max((1 - f[mask]).sum(), 1e-9))})
        return rows


def score_split(D, split, rdir, tune_folds=(1, 2), thrs=None, margins=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                drop=None):
    thrs = thrs or [round(x, 2) for x in np.arange(0.2, 0.97, 0.05)]
    sc = Scorer(split, drop)
    best = best_per_record(D)
    tune_mask = np.isin(sc.fold, list(tune_folds))
    val_mask = sc.fold == VAL_FOLD
    grid = []
    for thr in thrs:
        for mg in margins:
            m, _ = sc.metrics(decide_records(best, thr, mg), tune_mask)
            grid.append({"thr": thr, "margin": mg, "f05_tune": m["macro_f05"], "p": m["pair_precision"],
                         "r": m["pair_recall"]})
    bst = max(grid, key=lambda r: r["f05_tune"])
    pred = decide_records(best, bst["thr"], bst["margin"])
    val, f = sc.metrics(pred, val_mask)
    val.update({"thr": bst["thr"], "margin": bst["margin"]})
    # candidate recall ceiling on val
    cand_hit = D.filter(pl.col("y") == 1).select("a").to_numpy().ravel() if "y" in D.columns else None
    if cand_hit is not None:
        covered = np.bincount(cand_hit, minlength=sc.n1)
        val["candidate_recall"] = float(covered[val_mask].sum() / max(sc.n_true[val_mask].sum(), 1))
    # the same threshold scored on val over the grid (for the record only, not used to choose)
    val_curve = []
    for thr in thrs:
        m, _ = sc.metrics(decide_records(best, thr, bst["margin"]), val_mask)
        val_curve.append({"thr": thr, "f05_val": m["macro_f05"]})
    bd = sc.breakdown(f, val_mask)
    json.dump({"val": val, "best": bst, "grid": grid, "val_curve": val_curve, "breakdown": bd},
              open(os.path.join(rdir, "decision.json"), "w"), indent=1)
    pl.DataFrame(bd).write_csv(os.path.join(rdir, "val_breakdown.csv"))
    # error files on val
    s1, rr = ids(split)
    pv = pred.join(D.select(["a", "b", "p"]), on=["a", "b"], how="left")
    pv = pv.filter(pl.Series(val_mask[pv["a"].to_numpy()]))
    ya = sc.true_s1_of_b[pv["b"].to_numpy()]
    fp = pv.filter(pl.Series(ya != pv["a"].to_numpy())).with_columns(pl.Series("true_a", ya[ya != pv["a"].to_numpy()]))
    Tv = sc.T.filter(pl.Series(val_mask[sc.T["a"].to_numpy()]))
    fn = Tv.join(pred.with_columns(pl.lit(1).alias("hit")), on=["a", "b"], how="left").filter(pl.col("hit").is_null())
    fn = fn.join(D.select(["a", "b", "p"]), on=["a", "b"], how="left").join(best.rename({"a": "pred_a"}), on="b", how="left")

    def named(x, cols):
        s1id = s1["id"].to_numpy()
        rid = rr["id"].to_numpy()
        out = x
        for c in cols:
            arr = x[c].to_numpy()
            if c == "b":
                out = out.with_columns(pl.Series("rid", rid[arr]))
            else:
                v = np.where(arr >= 0, arr, 0)
                out = out.with_columns(pl.Series(f"{c}_id", np.where(arr >= 0, s1id[v], "")))
        return out
    named(fp, ["a", "b", "true_a"]).sort("p", descending=True).write_csv(os.path.join(rdir, "val_fp.tsv"), separator="\t")
    named(fn.with_columns(pl.col("pred_a").fill_null(-1)), ["a", "b", "pred_a"]).write_csv(
        os.path.join(rdir, "val_fn.tsv"), separator="\t")
    return {"val": val, "best": bst, "grid": grid}
