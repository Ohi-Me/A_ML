"""Which true pairs does the embedding list miss (top-K)? Writes a readable sample with the wrongly ranked S1."""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, load_source  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="e2")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()
gpu_init()
s1 = load_source("train", 1)
rr = pl.concat([load_source("train", 2), load_source("train", 3)])
n1 = s1.height
idx = np.load(os.path.join(CACHE, "emb", f"{args.tag}_train", "rid_emb_idx.npy"), mmap_mode="r")
pairs, _ = load_gt()
m1 = pl.DataFrame({"s1": s1["entity_id"], "a": np.arange(n1, dtype=np.int64)})
mr = pl.DataFrame({"rid": rr["entity_id"], "b": np.arange(rr.height, dtype=np.int64)})
g = pairs.join(m1, on="s1").join(mr, on="rid").filter(
    pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64) == VAL_FOLD)
a, b = g["a"].to_numpy(), g["b"].to_numpy()
L = np.asarray(idx[b])
hit = (L[:, :args.k] == a[:, None]).any(1)
miss = np.where(~hit)[0]
print("val pairs", len(a), "missed @", args.k, len(miss), flush=True)
rng = np.random.default_rng(0)
sel = rng.choice(miss, min(400, len(miss)), replace=False)
n1s, a1s, cs = s1["business_name"].to_numpy(), s1["business_address"].to_numpy(), s1["country"].to_numpy()
n2s, a2s = rr["business_name"].to_numpy(), rr["business_address"].to_numpy()
stats = Counter()
for i in miss:
    stats["empty_addr"] += a2s[b[i]].strip() in ("", '""')
    stats["nonlatin_name"] += any(ord(ch) > 0x24F for ch in n2s[b[i]])
    stats[cs[a[i]]] += 1
    stats["domain"] += (".com" in n2s[b[i]].lower() or ".c0m" in n2s[b[i]].lower())
out = os.path.join(RESULTS, "embeddings", args.tag, f"val_missed_top{args.k}.tsv")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("country\ts1_name\ts1_addr\trec_name\trec_addr\ttop1_name\ttop1_addr\n")
    for i in sel:
        t = L[i, 0]
        fh.write("\t".join([cs[a[i]], n1s[a[i]], a1s[a[i]], n2s[b[i]], a2s[b[i]], n1s[t] if t >= 0 else "",
                            a1s[t] if t >= 0 else ""]) + "\n")
print(dict(stats), flush=True)
