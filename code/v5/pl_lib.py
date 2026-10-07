"""Pseudo-labels for an unseen country (transductive self-training) and an XGBoost data iterator that mixes the
labelled training parts with pseudo-labelled rows.

Selection is deliberately conservative and uses evidence the pair model does not decide alone:
  positive record : its best S1 has calibrated p >= pos_thr, beats the runner-up by >= margin, AND the address agrees
                    (same first house number, or neither side has a number) with address-token jaccard >= addr_jac
                    -> the best pair is y=1, every other candidate of that record y=0 (a record matches one S1 at most)
  distractor record: best calibrated p <= neg_thr -> every candidate y=0 (teaches the unseen country's distractors)
Everything in between is left out. The same rule is validated on India -> US (loco_eval.py step pl) before it is
used for France.
"""
import numpy as np
import polars as pl
import xgboost as xgb

from common.xgbdata import read_part

AGREE_COLS = ["fnum_eq", "num_n1", "num_n2", "a_jac"]


def select_pseudo(D, F, pos_thr=0.97, margin=0.5, neg_thr=0.02, addr_jac=0.3, max_pairs=4_000_000, seed=0):
    """D: a, b, p (calibrated) for the candidate pairs of the target records; F: a, b + AGREE_COLS.
    Returns a, b, y (pseudo) and a small report."""
    s = D.sort(["b", "p"], descending=[False, True])
    best = s.group_by("b", maintain_order=True).agg(pl.col("a").first(), pl.col("p").first().alias("p1"),
                                                    pl.col("p").slice(1, 1).first().alias("p2")).with_columns(pl.col("p2").fill_null(0.0))
    best = best.join(F.select(["a", "b"] + AGREE_COLS), on=["a", "b"], how="left")
    agree = ((pl.col("fnum_eq") == 1) | ((pl.col("num_n1") == 0) & (pl.col("num_n2") == 0))) & (pl.col("a_jac") >= addr_jac)
    pos = best.filter((pl.col("p1") >= pos_thr) & ((pl.col("p1") - pl.col("p2")) >= margin) & agree.fill_null(False))
    neg = best.filter(pl.col("p1") <= neg_thr)
    rec = pl.concat([pos.select(["b", pl.col("a").cast(pl.Int64).alias("a_pos")]),
                     neg.select(["b", pl.lit(-1, dtype=pl.Int64).alias("a_pos")])])
    rows = D.select(["a", "b"]).join(rec, on="b", how="inner").with_columns(
        (pl.col("a").cast(pl.Int64) == pl.col("a_pos")).cast(pl.Int8).alias("y")).select(["a", "b", "y"])
    if rows.height > max_pairs:
        keep_b = rows.select("b").unique().sample(fraction=max_pairs / rows.height, seed=seed)
        rows = rows.join(keep_b, on="b", how="semi")
    rep = {"records": int(best.height), "pos_records": int(pos.height), "distractor_records": int(neg.height),
           "pairs": int(rows.height), "pos_pairs": int(rows["y"].sum()) if rows.height else 0}
    return rows, rep


class MixIter(xgb.DataIter):
    """Labelled rows of `files` (given folds, optional S1 mask) followed by pseudo-labelled rows of `extra_files`
    (inner join with `pl_rows` a, b, y; rows' own y is replaced). rename / override are applied to extra parts."""

    def __init__(self, files, feats, folds, a_mask=None, extra_files=(), pl_rows=None, rename=None, override=None):
        self.files, self.feats, self.folds, self.a_mask = list(files), feats, folds, a_mask
        self.extra = list(extra_files)
        self.pl_rows, self.rename, self.override = pl_rows, rename or {}, override
        self.i = 0
        self.rows = self.pos = self.pl_n = 0
        self.last = (0, 0)                  # (rows, pseudo rows) of the last complete pass
        super().__init__(cache_prefix=None)

    def _emit(self, input_data, d, y):
        X = d.select([pl.col(c).cast(pl.Float32) for c in self.feats]).to_numpy()
        self.rows += len(y)
        self.pos += int(y.sum())
        input_data(data=X, label=y.astype(np.float32))

    def next(self, input_data):
        while self.i < len(self.files) + len(self.extra):
            k = self.i
            self.i += 1
            if k < len(self.files):
                d = read_part(self.files[k], columns=self.feats + ["y", "fold", "a"]).filter(pl.col("fold").is_in(self.folds))
                if self.a_mask is not None and d.height:
                    d = d.filter(pl.Series(self.a_mask[d["a"].to_numpy()]))
                if d.height == 0:
                    continue
                self._emit(input_data, d, d["y"].to_numpy())
                return True
            d = pl.read_parquet(self.extra[k - len(self.files)])
            d = d.rename({o: n for o, n in self.rename.items() if o in d.columns})
            if self.override is not None:
                from phase2.p2lib import apply_override
                d = apply_override(d, self.override)
            d = d.drop("y", strict=False).join(self.pl_rows, on=["a", "b"], how="inner")
            if d.height == 0:
                continue
            self.pl_n += d.height
            self._emit(input_data, d, d["y"].to_numpy())
            return True
        return False

    def reset(self):                    # QuantileDMatrix reads the data more than once: count one pass only
        if self.rows:
            self.last = (self.rows, self.pl_n)
        self.i = 0
        self.rows = self.pos = self.pl_n = 0
