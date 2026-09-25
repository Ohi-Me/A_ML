"""Quick check of common.features on a slice of the candidate table: runs, speed, value ranges."""
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.features import compute_features_v2 as compute_features  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, NORM  # noqa: E402

gpu_init()
NCOLS = ["id", "country", "core", "key", "legal", "is_domain", "nonlatin", "am", "nums", "state", "first_num"]
C = pl.read_parquet(os.path.join(CACHE, "cands", "c1_train.parquet")).sample(500_000, seed=0)
S1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=NCOLS)
R = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=NCOLS) for k in (2, 3)])
t = time.time()
F = compute_features(S1[C["a"].to_numpy()], R[C["b"].to_numpy()])
dt = time.time() - t
print(f"{F.height} pairs in {dt:.1f}s -> {F.height / dt / 1e6:.2f} M pairs/s", flush=True)
pl.Config.set_tbl_cols(60)
pl.Config.set_tbl_rows(80)
print(F.describe().transpose(include_header=True, column_names="statistic").select(
    ["column", "mean", "min", "max", "null_count"]), flush=True)
