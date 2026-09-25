"""Compute pair features for a candidate table (all folds), with labels for the train split.
Streams to parquet parts so the 32 GB job memory cap is never hit.

  python code/tfidf_fuzzy/pair_features.py --cands c1 --split train --btag b1 --etags e1

Output: data/cache/feats/<cands>_<split>/part_XXX.parquet (a, b, fold, y, features...)."""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.features import compute_features, compute_features_v2  # noqa: E402
from common.split import fold_of  # noqa: E402

NCOLS = ["id", "country", "core", "key", "legal", "is_domain", "nonlatin", "am", "nums", "state", "first_num"]


def rowdot(M, a, b, off):
    return np.asarray(M[a].multiply(M[b + off]).sum(axis=1)).ravel().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--norm", default=NORM)
    ap.add_argument("--etags", default="e1", help="embedding runs whose cosine is added as a feature")
    ap.add_argument("--fset", default="v1", choices=["v1", "v2"])
    ap.add_argument("--pairc", default="", help="parquet with a, b, cos_<tag> from the OOF bi-encoder (joined in)")
    ap.add_argument("--part", type=int, default=2_000_000, help="pairs per parquet part")
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    out_dir = os.path.join(CACHE, "feats", f"{args.cands}_{args.split}")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    bdir = os.path.join(CACHE, "blocking", f"{args.btag}_{args.split}")
    # "src:name" reads embeddings of run src but names the feature cos_<name> (test uses the all-folds model)
    etag_pairs = [(x.split(":")[0], x.split(":")[-1]) for x in args.etags.split(",") if x]
    etags = [n for _, n in etag_pairs]

    C = pl.read_parquet(os.path.join(CACHE, "cands", f"{args.cands}_{args.split}.parquet")).sort(["b", "a"])
    S1 = pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s1.parquet"), columns=NCOLS)
    R = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s{k}.parquet"), columns=NCOLS)
                   for k in (2, 3)])
    n1 = S1.height
    a_all = C["a"].to_numpy()
    b_all = C["b"].to_numpy()
    print("pairs", C.height, "S1", n1, "records", R.height, flush=True)

    with Timer("tfidf cosines"):
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        cn = np.zeros(C.height, np.float32)
        ca = np.zeros(C.height, np.float32)
        for s in range(0, C.height, 2_000_000):
            e = min(s + 2_000_000, C.height)
            cn[s:e] = rowdot(NAME, a_all[s:e], b_all[s:e], n1)
            ca[s:e] = rowdot(ADDR, a_all[s:e], b_all[s:e], n1)
        del NAME, ADDR
    C = C.with_columns(pl.Series("cos_name", cn), pl.Series("cos_addr", ca), pl.Series("cos_sum", cn + ca))
    del cn, ca
    import torch
    for src, et in etag_pairs:
        with Timer(f"embedding cosine {src} -> cos_{et} (GPU)"):
            E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", f"{src}_{args.split}", "emb.npy"))).cuda()
            ce = np.zeros(C.height, np.float32)
            for s in range(0, C.height, 4_000_000):
                e = min(s + 4_000_000, C.height)
                ia = torch.from_numpy(a_all[s:e].astype(np.int64)).cuda()
                ib = torch.from_numpy(b_all[s:e].astype(np.int64) + n1).cuda()
                ce[s:e] = (E[ia].float() * E[ib].float()).sum(1).cpu().numpy()
            del E
            torch.cuda.empty_cache()
            C = C.with_columns(pl.Series(f"cos_{et}", ce))

    if args.pairc:
        with Timer("join OOF pair cosines"):
            PC = pl.read_parquet(os.path.join(CACHE, args.pairc))
            pc_col = [c for c in PC.columns if c.startswith("cos_")][0]
            C = C.join(PC, on=["a", "b"], how="left").with_columns(pl.col(pc_col).fill_null(0.0))
            etags = etags + [pc_col[4:]]
    with Timer("context features"):
        # competition among the S1 candidates of one record, and among the records of one S1
        ctx_cols = ["cos_sum"] + [f"cos_{et}" for et in etags]
        exprs = [pl.len().over("b").cast(pl.Int32).alias("b_n"), pl.len().over("a").cast(pl.Int32).alias("a_n")]
        for c in ctx_cols:
            exprs += [
                (pl.col(c) - pl.col(c).max().over("b")).alias(f"b_gap_{c}"),
                pl.col(c).rank("ordinal", descending=True).over("b").cast(pl.Int32).alias(f"b_rank_{c}"),
                (pl.col(c) - pl.col(c).max().over("a")).alias(f"a_gap_{c}"),
                pl.col(c).rank("ordinal", descending=True).over("a").cast(pl.Int32).alias(f"a_rank_{c}"),
            ]
        C = C.with_columns(exprs)
        for c in ctx_cols:
            # margin over the runner-up S1 of the same record (0 for the only candidate)
            sec = C.filter(pl.col(f"b_rank_{c}") == 2).select(["b", pl.col(c).alias("_sec")])
            C = C.join(sec, on="b", how="left").with_columns(
                (pl.col(c) - pl.col("_sec").fill_null(0.0)).alias(f"b_gap2_{c}")).drop("_sec")
        kf1 = S1.group_by(["country", "key"]).len().rename({"len": "key_n_s1"})
        kfr = R.group_by(["country", "key"]).len().rename({"len": "key_n_r"})
        ka = S1.select(["country", "key"]).with_row_index("a").with_columns(pl.col("a").cast(pl.Int32)) \
            .join(kf1, on=["country", "key"], how="left").select(["a", "key_n_s1"])
        kb = R.select(["country", "key"]).with_row_index("b").with_columns(pl.col("b").cast(pl.Int32)) \
            .join(kfr, on=["country", "key"], how="left").select(["b", "key_n_r"])
        C = C.join(ka, on="a", how="left").join(kb, on="b", how="left")

    with Timer("labels"):
        fold_arr = np.array([fold_of(x) for x in S1["id"].to_list()], np.int8)
        C = C.with_columns(pl.Series("fold", fold_arr[C["a"].to_numpy()]))
        if args.split == "train":
            pairs, _ = load_gt()
            s1_map = pl.DataFrame({"s1": S1["id"], "a": np.arange(n1, dtype=np.int32)})
            r_map = pl.DataFrame({"rid": R["id"], "b": np.arange(R.height, dtype=np.int32)})
            g = pairs.join(s1_map, on="s1").join(r_map, on="rid").select(["a", "b"]).with_columns(pl.lit(1, pl.Int8).alias("y"))
            C = C.join(g, on=["a", "b"], how="left").with_columns(pl.col("y").fill_null(0))
            has = g.select("b").unique().with_columns(pl.lit(1, pl.Int8).alias("b_has_match"))
            C = C.join(has, on="b", how="left").with_columns(pl.col("b_has_match").fill_null(0))
        C = C.sort(["b", "a"])
        a_all = C["a"].to_numpy()
        b_all = C["b"].to_numpy()

    with Timer("string features (streamed parts)"):
        for pi, ps in enumerate(range(0, C.height, args.part)):
            pe = min(ps + args.part, C.height)
            FE = (compute_features_v2 if args.fset == "v2" else compute_features)(S1[a_all[ps:pe]], R[b_all[ps:pe]])
            pl.concat([C[ps:pe], FE], how="horizontal").write_parquet(os.path.join(out_dir, f"part_{pi:03d}.parquet"))
            print(f"  part {pi}: pairs {ps}-{pe}, {time.time() - t0:.0f}s", flush=True)
            del FE
    rep = {"pairs": C.height, "cols": C.columns, "runtime_s": time.time() - t0}
    if "y" in C.columns:
        rep["positives"] = int(C["y"].sum())
    print(json.dumps(rep), flush=True)
    os.makedirs(os.path.join(RESULTS, "tfidf_fuzzy"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "tfidf_fuzzy", f"features_{args.cands}_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
