"""Where do the remaining errors sit on the probability scale? (val fold, test-prior simulation)
Borderline errors can be fixed by a better model; confident errors are mostly ambiguity / duplicates."""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.split import VAL_FOLD  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "xgb_c2_sim19"
gpu_init()
D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{tag}.parquet"))
sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
best = best_per_record(D)
tb = sc.true_s1_of_b[best["b"].to_numpy()]
a = best["a"].to_numpy()
val_rec = np.zeros(len(a), bool)
# records whose best S1 or true S1 is in the val fold
val_rec |= sc.fold[a] == VAL_FOLD
val_rec |= (tb >= 0) & (sc.fold[np.maximum(tb, 0)] == VAL_FOLD)
correct = tb == a
p1 = best["p1"].to_numpy()
bins = [0, 0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.95, 0.99, 1.0001]
out = {"tag": tag, "bins": bins, "correct_best": [], "wrong_best": []}
for lo, hi in zip(bins[:-1], bins[1:]):
    m = val_rec & (p1 >= lo) & (p1 < hi)
    out["correct_best"].append(int((m & correct).sum()))
    out["wrong_best"].append(int((m & ~correct).sum()))
print(json.dumps(out), flush=True)
json.dump(out, open(os.path.join(RESULTS, "xgboost", tag, "headroom.json"), "w"), indent=1)
