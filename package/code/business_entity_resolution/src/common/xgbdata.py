"""Stream feature parquet parts into XGBoost without holding everything in host RAM."""
import numpy as np
import polars as pl
import xgboost as xgb


def read_part(f, columns=None):
    """f: a parquet path, or (main, sidecar) whose rows line up; returns one frame with the wanted columns."""
    if isinstance(f, str):
        return pl.read_parquet(f, columns=columns)
    main, side = f
    if columns is None:
        return pl.concat([pl.read_parquet(main), pl.read_parquet(side)], how="horizontal")
    sc = set(pl.read_parquet_schema(side).keys())
    mc = [c for c in columns if c not in sc]
    d = pl.read_parquet(main, columns=mc)
    sd = [c for c in columns if c in sc]
    return pl.concat([d, pl.read_parquet(side, columns=sd)], how="horizontal") if sd else d


def pair_files(main_files, side_dir):
    import glob
    import os
    side = sorted(glob.glob(os.path.join(side_dir, "part_*.parquet")))
    assert len(side) == len(main_files), (len(side), len(main_files))
    return list(zip(main_files, side))


class PartIter(xgb.DataIter):
    """Streams parquet parts (rows of the given folds) into a QuantileDMatrix: host RAM holds one part at a time."""

    def __init__(self, files, feats, folds, neg_keep=1.0, seed=0, a_mask=None):
        self.files, self.feats, self.folds, self.neg_keep, self.seed = files, feats, folds, neg_keep, seed
        self.a_mask = a_mask     # optional bool array over S1 rows: keep only pairs of these S1
        self.i = 0
        self.rows = 0
        self.pos = 0
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        while self.i < len(self.files):
            d = read_part(self.files[self.i], columns=self.feats + ["y", "fold", "a"]).filter(
                pl.col("fold").is_in(self.folds))
            if self.a_mask is not None and d.height:
                d = d.filter(pl.Series(self.a_mask[d["a"].to_numpy()]))
            self.i += 1
            if self.neg_keep < 1.0 and d.height:
                rng = np.random.default_rng(self.seed * 1000 + self.i)
                d = d.filter(pl.Series((d["y"].to_numpy() == 1) | (rng.random(d.height) < self.neg_keep)))
            if d.height == 0:
                continue
            X = d.select([pl.col(c).cast(pl.Float32) for c in self.feats]).to_numpy()
            y = d["y"].to_numpy().astype(np.float32)
            self.rows += len(y)
            self.pos += int(y.sum())
            input_data(data=X, label=y)
            return True
        return False

    def reset(self):
        self.i = 0
