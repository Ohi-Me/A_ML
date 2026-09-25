"""Score a list of pairs with the fine-tuned multilingual cross-encoder halves, keeping each half separate.

Used by the chain-consistent runs: chain A needs cross-encoder A on the pairs that stage-1 A finds uncertain, chain B
needs B on the pairs B finds uncertain. Input: a parquet under data/cache with a, b and stage-1 columns (pA, pB, ...).
--jobs lists output=half@stage1column: the pair is scored by that half when the stage-1 column is inside the band.

  python code/neural_reranker/xenc_score_pairs.py --split test --pairs feats/test_chain_p1.parquet \
      --jobs xA=A@pA,xB=B@pB --out xenc/chain_test.parquet
Output: a, b and one column per job (null where the pair was not in that job's band).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, load_source  # noqa: E402
from neural_reranker.ml_cross_encoder import PairModel, score, tokenize  # noqa: E402
from neural_reranker.prep_pairs import txt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="hardv2", help="cross-encoder run (models xenc/ml_<name>_model_<half>.pt)")
    ap.add_argument("--jobs", default="xA=A@pA,xB=B@pB")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--maxlen", type=int, default=110)
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    jobs = []
    for j in args.jobs.split(","):
        o, rest = j.split("=")
        h, pc = rest.split("@")
        jobs.append((o, h, pc))
    D = pl.read_parquet(os.path.join(CACHE, args.pairs))
    D = D.with_columns([((pl.col(pc) >= args.lo) & (pl.col(pc) <= args.hi)).alias(f"in_{o}") for o, h, pc in jobs])
    U = D.filter(pl.any_horizontal([pl.col(f"in_{o}") for o, _, _ in jobs]))
    rep = {"args": vars(args), "pairs": D.height, "union_band": U.height,
           **{f"band_{o}": int(D[f"in_{o}"].sum()) for o, _, _ in jobs}}
    print(json.dumps(rep), flush=True)
    del D
    with Timer("texts + tokenize"):
        s1 = load_source(args.split, 1)
        rr = pl.concat([load_source(args.split, 2), load_source(args.split, 3)])
        a, b = U["a"].to_numpy(), U["b"].to_numpy()
        t1 = txt(s1["business_name"].to_numpy()[a], s1["business_address"].to_numpy()[a], args.maxlen, True)
        t2 = txt(rr["business_name"].to_numpy()[b], rr["business_address"].to_numpy()[b], args.maxlen, True)
        del s1, rr
        ids, pad = tokenize(t1, t2)
        del t1, t2
    out = U.select(["a", "b"])
    for h in sorted(set(h for _, h, _ in jobs)):
        mine = [o for o, hh, _ in jobs if hh == h]
        need = np.zeros(U.height, bool)
        for o in mine:
            need |= U[f"in_{o}"].to_numpy()
        sel = np.where(need)[0]
        full = np.full(U.height, np.nan, np.float32)
        with Timer(f"cross-encoder {h}: {len(sel)} pairs"):
            m = PairModel().to(dev)
            m.load_state_dict(torch.load(os.path.join(CACHE, "xenc", f"ml_{args.name}_model_{h}.pt"), map_location=dev))
            full[sel] = score(m, ids[sel], pad, dev)
            del m
            torch.cuda.empty_cache()
        for o in mine:
            col = np.where(U[f"in_{o}"].to_numpy(), full, np.nan).astype(np.float32)
            out = out.with_columns(pl.Series(o, col).fill_nan(None))
    out.write_parquet(os.path.join(CACHE, args.out))
    rep.update({"runtime_s": time.time() - t0, "gpu": gpu_mem()})
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
