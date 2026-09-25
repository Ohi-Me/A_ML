"""Readable error analysis for a pair-model run: texts + key features of false positives / false negatives on val.

  python code/xgboost/error_analysis.py --tag xgb_c1 --feats c1_train
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_source  # noqa: E402

KEYF = ["cos_e2", "cos_name", "cos_addr", "b_gap2_cos_e2", "n_tset", "a_tset", "num_conflict", "fnum_eq", "a_empty2",
        "legal_jac", "key_n_s1", "state_cmp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--sub", default="xgboost")
    args = ap.parse_args()
    gpu_init()
    rdir = os.path.join(RESULTS, args.sub, args.tag)
    s1 = load_source("train", 1)
    rr = pl.concat([load_source("train", 2), load_source("train", 3)])
    n1s, a1s, cs = s1["business_name"].to_numpy(), s1["business_address"].to_numpy(), s1["country"].to_numpy()
    n2s, a2s, rid = rr["business_name"].to_numpy(), rr["business_address"].to_numpy(), rr["entity_id"].to_numpy()
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    cols = pl.scan_parquet(files).collect_schema().names()
    keyf = [c for c in KEYF if c in cols]
    fp = pl.read_csv(os.path.join(rdir, "val_fp.tsv"), separator="\t")
    fn = pl.read_csv(os.path.join(rdir, "val_fn.tsv"), separator="\t")
    want = pl.concat([fp.select(["a", "b"]), fn.select(["a", "b"]),
                      fp.filter(pl.col("true_a") >= 0).select([pl.col("true_a").alias("a"), "b"]),
                      fn.filter(pl.col("pred_a") >= 0).select([pl.col("pred_a").alias("a"), "b"])]).unique()
    F = pl.scan_parquet(files).select(["a", "b"] + keyf).join(want.lazy(), on=["a", "b"]).collect()
    rep = {}
    # ---- false positives
    fpj = fp.join(F, on=["a", "b"], how="left")
    cat = Counter()
    lines = []
    for row in fpj.iter_rows(named=True):
        a, b, ta = row["a"], row["b"], row["true_a"]
        kind = "distractor record (no true S1)" if ta < 0 else "record belongs to another S1"
        if row.get("a_empty2") == 1:
            kind += " | empty address"
        cat[kind] += 1
        if len(lines) < 400:
            t = f"   TRUE S1: {n1s[ta]} | {a1s[ta]}" if ta >= 0 else "   TRUE S1: none"
            lines.append(f"[{cs[a]}] p={row['p']:.3f} {kind}\n   PRED S1: {n1s[a]} | {a1s[a]}\n   RECORD : {n2s[b]} | {a2s[b]}\n{t}\n"
                         f"   feats: " + ", ".join(f"{k}={row[k]:.2f}" for k in keyf if row.get(k) is not None))
    rep["fp_categories"] = dict(cat)
    open(os.path.join(rdir, "err_fp_readable.txt"), "w", encoding="utf-8").write("\n".join(lines))
    # ---- false negatives
    fnj = fn.join(F, on=["a", "b"], how="left")
    cat = Counter()
    lines = []
    for row in fnj.iter_rows(named=True):
        a, b, pa = row["a"], row["b"], row["pred_a"]
        if row["p"] is None:
            kind = "blocking miss (not a candidate)"
        elif pa >= 0 and pa != a:
            kind = "lost to another S1" + (" (other accepted)" if row["p1"] is not None and row["p1"] >= 0.5 else "")
        else:
            kind = "own best but below threshold/margin"
        if a2s[b].strip() in ("", '""'):
            kind += " | empty address"
        elif any(ord(ch) > 0x24F for ch in n2s[b]):
            kind += " | native script name"
        cat[kind] += 1
        if len(lines) < 500 and row["p"] is not None:
            o = f"   BEST S1: {n1s[pa]} | {a1s[pa]} (p1={row['p1']:.3f})" if pa >= 0 and pa != a else ""
            lines.append(f"[{cs[a]}] p={row['p']:.3f} {kind}\n   TRUE S1: {n1s[a]} | {a1s[a]}\n   RECORD : {n2s[b]} | {a2s[b]}\n{o}\n"
                         f"   feats: " + ", ".join(f"{k}={row[k]:.2f}" for k in keyf if row.get(k) is not None))
    rep["fn_categories"] = dict(cat)
    open(os.path.join(rdir, "err_fn_readable.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(os.path.join(rdir, "error_categories.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
