"""Turn blocking lists into the candidate pair table.

--lists is a comma list of DIR:LIST:K. DIR is a folder under data/cache/ that holds rid_<LIST>_idx.npy
(for example blocking/b1_train or emb/e1_train); K is the depth kept from that list. A pair (S1 row a, record row b)
is a candidate if b's list has a within the first K places, for any list. Ranks are kept as features (99 = absent).

  python code/blocking/build_candidates.py --name c1 --split train \
      --lists blocking/b1:name:5,blocking/b1:addr:5,blocking/b1:comb:10,emb/e1:emb:10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--lists", required=True)
    ap.add_argument("--norm", default=NORM)
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    out_path = os.path.join(CACHE, "cands", f"{args.name}_{args.split}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rdir = os.path.join(RESULTS, "blocking", "candidates")
    os.makedirs(rdir, exist_ok=True)

    specs = []
    for item in args.lists.split(","):
        d, l, k = item.split(":")
        specs.append((d, l, int(k)))
    parts, rank_cols = [], []
    with Timer("load lists"):
        for d, l, k in specs:
            col = f"r_{os.path.basename(d)}_{l}"
            rank_cols.append(col)
            idx = np.load(os.path.join(CACHE, f"{d}_{args.split}", f"rid_{l}_idx.npy"), mmap_mode="r")[:, :k]
            nr = idx.shape[0]
            f = pl.DataFrame({"a": np.ascontiguousarray(idx).ravel(), "b": np.repeat(np.arange(nr, dtype=np.int32), k),
                              col: np.tile(np.arange(k, dtype=np.int8), nr)}).filter(pl.col("a") >= 0)
            parts.append(f)
    with Timer("union"):
        allp = pl.concat(parts, how="diagonal")
        del parts
        C = allp.group_by(["a", "b"]).agg([pl.col(c).min() for c in rank_cols])
        del allp
        C = C.with_columns([pl.col(c).fill_null(99).cast(pl.Int8) for c in rank_cols]).sort(["b", "a"])
    C.write_parquet(out_path)
    print("pairs", C.height, flush=True)

    rep = {"name": args.name, "lists": args.lists, "pairs": C.height, "split": args.split}
    s1_ids = pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s1.parquet"), columns=["id"])["id"]
    rep["n_s1"] = len(s1_ids)
    rep["s1_with_candidates"] = int(C["a"].n_unique())
    rep["records_with_candidates"] = int(C["b"].n_unique())
    rep["cands_per_record_mean"] = C.height / max(rep["records_with_candidates"], 1)
    if args.split == "train":
        with Timer("candidate recall"):
            rid_ids = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s{k}.parquet"),
                                                 columns=["id"]) for k in (2, 3)])["id"]
            pairs, per = load_gt()
            s1_map = pl.DataFrame({"s1": s1_ids, "a": np.arange(len(s1_ids), dtype=np.int32)})
            r_map = pl.DataFrame({"rid": rid_ids, "b": np.arange(len(rid_ids), dtype=np.int32)})
            g = pairs.join(s1_map, on="s1").join(r_map, on="rid").with_columns(
                pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
            hit = g.join(C.select(["a", "b"]).with_columns(pl.lit(True).alias("hit")), on=["a", "b"], how="left") \
                .with_columns(pl.col("hit").fill_null(False))
            for nm, h in (("val", hit.filter(pl.col("fold") == VAL_FOLD)), ("all", hit)):
                rep[f"cand_recall_{nm}"] = float(h["hit"].mean())
            # recall of each list alone at its depth, on val
            hv = hit.filter(pl.col("fold") == VAL_FOLD).join(C, on=["a", "b"], how="left")
            for c in rank_cols:
                rep[f"val_recall_{c}"] = float((hv[c].fill_null(99) < 99).mean())
            vs = per.with_columns(pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold")) \
                .filter(pl.col("fold") == VAL_FOLD).join(s1_map, on="s1")
            cnt = C.group_by("a").len()
            vc = vs.join(cnt, on="a", how="left").with_columns(pl.col("len").fill_null(0))
            rep["val_s1_cands_mean"] = float(vc["len"].mean())
            rep["val_s1_cands_p50_p90_p99_max"] = [float(vc["len"].quantile(q)) for q in (0.5, 0.9, 0.99)] + [int(vc["len"].max())]
            full = hit.filter(pl.col("fold") == VAL_FOLD).group_by("s1").agg(pl.col("hit").all())
            rep["val_s1_all_matches_covered"] = float(full["hit"].mean())
            hit.filter((pl.col("fold") == VAL_FOLD) & ~pl.col("hit")).select(["s1", "rid"]).head(5000).write_csv(
                os.path.join(rdir, f"missed_{args.name}.tsv"), separator="\t")
    rep["runtime_s"] = time.time() - t0
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(os.path.join(rdir, f"cands_{args.name}_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
