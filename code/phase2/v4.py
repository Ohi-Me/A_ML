"""V4 candidates: replay on test exactly the chain that was validated, instead of a refit / an average of models.

Principle (from the V3 audit): every score a downstream model sees on test must be produced the same way as on the
validation fold. Chain A = the models that scored fold 0 (and fold 4) in validation:
  stage-1 half A (full + no-embedding blend) -> XLM-R half A on the pairs in A's own band -> GNN half A
with the bi-encoder cosine computed the validation way (one held-out fold model per S1: 'hash1').
Its validation score is therefore exactly V3's / V2's fold-0 score, and the test score is its faithful replay.

Candidates
  V4A  corrected V3   : chain A with XLM-R and GNN (V3 checkpoints gnn_mlx_v2_A, XLM-R hardv2 A)
  V4B  simplified V3  : chain A, GNN without XLM-R (gnn_v2_A)
  V4C  V2-style       : chain A, stage 2 XGBoost (xgb_c2_blend_s2_v2 model A) on stage-1 half A
Averaging halves / refitting is deliberately avoided: both change the score distribution the calibration was fitted on.

Steps (data/cache_v2 relative):
  python code/phase2/cos_variants.py stage1 --var hash1 --models halfA,refit
  python code/phase2/v4.py prep                 # pairs tables for the cross-encoder scorer
  python code/neural_reranker/xenc_score_pairs.py --split test --pairs p2/v4_pairs_halfA_hash1.parquet --jobs xA=A@pA --out p2/xenc_chainA_hash1.parquet
  python code/neural_reranker/xenc_score_pairs.py --split test --pairs p2/v4_pairs_refit_hash1.parquet --jobs xA=A@pR,xB=B@pR --out p2/xenc_refit_hash1.parquet
  python code/phase2/v4.py xmean                # -> mlxenc columns
  python code/phase2/gnn_run.py score ...       # see run_phase2.sh
  python code/phase2/v4.py stage2               # V4C test scores
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import ids  # noqa: E402
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from phase2.p2lib import apply_override, cpath, device, dump  # noqa: E402

P2C = os.path.join(CACHE, "p2")


def cmd_prep(args):
    os.makedirs(P2C, exist_ok=True)
    for kind, col in (("halfA", "pA"), ("refit", "pR")):
        src = cpath(f"feats/p2_test_s1_{kind}_hash1.parquet")
        if os.path.exists(src):
            pl.read_parquet(src).rename({"p": col}).write_parquet(os.path.join(P2C, f"v4_pairs_{kind}_hash1.parquet"))
            print("wrote pairs table", kind, flush=True)


def cmd_xmean(args):
    """cross-encoder outputs (xA / xB, null outside the band) -> mlxenc = mean of the available halves."""
    for name in ("xenc_chainA_hash1", "xenc_refit_hash1"):
        p = os.path.join(P2C, f"{name}.parquet")
        if not os.path.exists(p):
            continue
        X = pl.read_parquet(p)
        xs = [c for c in ("xA", "xB") if c in X.columns]
        X = X.with_columns(pl.mean_horizontal([pl.col(c) for c in xs]).alias("mlxenc")).filter(pl.col("mlxenc").is_not_null())
        X.select(["a", "b", "mlxenc"]).write_parquet(os.path.join(P2C, f"{name}_mlx.parquet"))
        print(name, "band pairs", X.height, flush=True)


def cmd_stage2(args):
    """V4C: V2's stage-2 model A on test, fed by stage-1 half A (hash1 cosine) - the way fold 0 was scored."""
    import scipy.sparse as sp
    import torch
    import xgboost as xgb
    from common.stage2_feats import context_features, sibling_features
    dev = device()
    t0 = time.time()
    tag = "xgb_c2_blend_s2_v2"
    rep2 = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
    f2 = rep2["features"]
    m2 = xgb.Booster(model_file=os.path.join(RESULTS, "xgboost", tag, "model_A.json"))
    D = pl.read_parquet(cpath(f"feats/p2_test_s1_halfA_{args.var}.parquet")).with_row_index("r")
    with Timer("stage-2 context + sibling features (test, stage-1 half A)"):
        D = context_features(D)
        E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", "e2f_test", "emb.npy"))).to(dev)
        bdir = os.path.join(CACHE, "blocking", "b1_test")
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        D = sibling_features(D, E, NAME, ADDR, ids("test")[0].height, dev).sort("r")
        del E, NAME, ADDR
        new_cols = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
    ov = pl.read_parquet(cpath(f"p2/cosvar/c2_test_{args.var}.parquet")) if args.var != "mean5" else None
    outs, off = [], 0
    with Timer("stage-2 model A: score test"):
        for f in sorted(glob.glob(os.path.join(CACHE, "feats", "c2_test", "part_*.parquet"))):
            d = pl.read_parquet(f).rename({"r_e2f_emb": "r_e3_emb"}, strict=False)
            d = apply_override(d, ov)
            add = D[off:off + d.height]
            assert (add["b"].to_numpy() == d["b"].to_numpy()).all() and (add["a"].to_numpy() == d["a"].to_numpy()).all()
            off += d.height
            d = pl.concat([d, add.select(new_cols).rename({"p": "s1_p"})], how="horizontal")
            X = d.select([pl.col(c).cast(pl.Float32) for c in f2]).to_numpy()
            outs.append(d.select(["a", "b"]).with_columns(pl.Series("p", m2.predict(xgb.DMatrix(X)))))
    pl.concat(outs).write_parquet(cpath(f"feats/p2_test_v4c_{args.var}_scores.parquet"))
    dump({"runtime_s": time.time() - t0, "var": args.var}, "v4", f"stage2_v4c_{args.var}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "xmean", "stage2"])
    ap.add_argument("--var", default="hash1")
    args = ap.parse_args()
    {"prep": cmd_prep, "xmean": cmd_xmean, "stage2": cmd_stage2}[args.step](args)


if __name__ == "__main__":
    main()
