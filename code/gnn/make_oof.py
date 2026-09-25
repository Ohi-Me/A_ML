"""Turn GNN edge logits into an out-of-fold score table (a, b, fold, y, p) so the usual decoders can use it."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE  # noqa: E402

base, scores, out = sys.argv[1], sys.argv[2], sys.argv[3]
gpu_init()
D = pl.read_parquet(os.path.join(CACHE, "feats", base)).select(["a", "b", "fold", "y"])
S = pl.read_parquet(os.path.join(CACHE, scores))
D = D.join(S, on=["a", "b"], how="left").with_columns((1 / (1 + (-pl.col("gnn")).exp())).cast(pl.Float32).alias("p")).drop("gnn")
D.write_parquet(os.path.join(CACHE, "feats", f"oof_{out}.parquet"))
print(D.height, D["p"].null_count(), float(D["p"].mean()))
