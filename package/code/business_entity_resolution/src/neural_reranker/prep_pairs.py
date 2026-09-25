"""Neural reranker, step 1: collect the uncertain pairs and their raw texts (ascii-transliterated, lower case).

Uncertain = stage-1 out-of-fold probability in [lo, hi]. Written to data/cache/xenc/<name>_<split>.parquet with
a, b, fold, y (train), p (stage-1) and the two texts "name | address"."""
import argparse
import os
import sys

import polars as pl
from anyascii import anyascii

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, load_source  # noqa: E402


def txt(names, addrs, n=96, raw=False):
    out = []
    for nm, ad in zip(names, addrs):
        s = f"{nm} | {ad}".replace('"', "")
        s = " ".join(s.split()) if raw else anyascii(s).lower()
        out.append(s[:n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--scores", required=True, help="parquet under data/cache/feats with a, b, p (+fold, y)")
    ap.add_argument("--name", default="hard")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--raw", action="store_true", help="keep original scripts / accents (pretrained multilingual model)")
    ap.add_argument("--maxlen", type=int, default=96)
    args = ap.parse_args()
    gpu_init()
    os.makedirs(os.path.join(CACHE, "xenc"), exist_ok=True)
    with Timer("select uncertain pairs"):
        D = pl.read_parquet(os.path.join(CACHE, "feats", args.scores))
        H = D.filter((pl.col("p") >= args.lo) & (pl.col("p") <= args.hi))
        print("pairs", D.height, "uncertain", H.height, flush=True)
    with Timer("texts"):
        s1 = load_source(args.split, 1)
        rr = pl.concat([load_source(args.split, 2), load_source(args.split, 3)])
        a, b = H["a"].to_numpy(), H["b"].to_numpy()
        t1 = txt(s1["business_name"].to_numpy()[a], s1["business_address"].to_numpy()[a], args.maxlen, args.raw)
        t2 = txt(rr["business_name"].to_numpy()[b], rr["business_address"].to_numpy()[b], args.maxlen, args.raw)
        H = H.with_columns(pl.Series("t1", t1), pl.Series("t2", t2))
    H.write_parquet(os.path.join(CACHE, "xenc", f"{args.name}_{args.split}.parquet"))
    if "y" in H.columns:
        print("positives", int(H["y"].sum()), "by fold", H.group_by("fold").len().sort("fold").to_dicts(), flush=True)


if __name__ == "__main__":
    main()
