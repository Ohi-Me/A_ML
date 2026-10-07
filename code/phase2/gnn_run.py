"""Train or score edge-GNN variants (phase 2). Wraps phase2/gnn_lib.py; V3's own checkpoints can be scored too.

  # ablation 6: V3 GNN without the bi-encoder cosine edge input (cross-fitted exactly like V3)
  python code/phase2/gnn_run.py train --tag gnn_mlx_nocos --extra xenc/ml_hardv2_train_scores.parquet:mlxenc --drop_feats cos_e3

  # ablation 7: V3's checkpoints on test, cosine computed the validation way (stage 1 / cross-encoder unchanged)
  python code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --split test --s1 test_stage1_final_v2.parquet \
      --extra xenc/ml_hardv2_test_scores.parquet:mlxenc --override hash1 --out v3_gnnhash1

Checkpoints: --ckpt is a prefix under the cache: gnn/gnn_mlx_v2 -> gnn/gnn_mlx_v2_A.pt, _B.pt (V3, edge_gnn.py), or
p2/gnn/<tag> for models trained here. Output scores: p2/gnn/<name>_<split>_scores.parquet (a, b, gnn logit).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import CACHE  # noqa: E402
from gnn.edge_gnn import FEATS  # noqa: E402
from phase2 import gnn_lib as GL  # noqa: E402
from phase2.p2lib import cpath, device, dump  # noqa: E402

GDIR = os.path.join(CACHE, "p2", "gnn")


def parse_extra(s):
    return [tuple(x.split(":")) for x in s.split(",") if x]


def override_frame(s):
    if not s:
        return None
    p = s if s.endswith(".parquet") else os.path.join("p2", "cosvar", f"c2_test_{s}.parquet")
    return pl.read_parquet(cpath(p))


def cmd_train(args, dev):
    from sklearn.metrics import average_precision_score, roc_auc_score
    t0 = time.time()
    feats = [f for f in FEATS if f not in set(args.drop_feats.split(","))]
    G = GL.build_graph(args.feats, args.s1, dev, parse_extra(args.extra), feats)
    fit_mask = None
    if args.fit_country:
        from common.decide import ids
        fit_mask = ids("train")[0]["country"].to_numpy() == args.fit_country
    lg = GL.crossfit(G, os.path.join(GDIR, args.tag), fit_mask, args.epochs, args.batch_s1, args.lr)
    out = G.M.select(["a", "b"]).with_columns(pl.Series("gnn", lg))
    out.write_parquet(os.path.join(GDIR, f"{args.tag}_train_scores.parquet"))
    G.M.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", (1 / (1 + np.exp(-lg))).astype(np.float32))) \
        .write_parquet(os.path.join(CACHE, "feats", f"oof_p2_{args.tag}.parquet"))
    v = G.fold_e == 0
    y = G.M["y"].to_numpy()
    rep = {"args": vars(args), "feats": feats, "val_auc": float(roc_auc_score(y[v], lg[v])),
           "val_ap": float(average_precision_score(y[v], lg[v])), "runtime_s": time.time() - t0}
    print(json.dumps(rep), flush=True)
    dump(rep, "gnn", f"{args.tag}_train.json")


def cmd_score(args, dev):
    t0 = time.time()
    halves = args.halves.split(",")
    paths = [cpath(f"{args.ckpt}_{h}.pt") for h in halves]
    ck = GL.load(paths[0], "cpu")
    feats = ck.get("feats", FEATS)
    G = GL.build_graph(args.feats, args.s1, dev, parse_extra(args.extra), feats, override_frame(args.override),
                       mu_sd=(ck["mu"].numpy(), ck["sd"].numpy()), with_labels=args.split == "train")
    nodes = None
    if args.split == "train":
        sf = GL.s1_folds(G)
        nodes = G.present[np.isin(sf[G.present], [int(x) for x in args.score_folds.split(",")])]
    lg = GL.score_with(G, paths, nodes, args.batch_s1)
    out = G.M.select(["a", "b"]).with_columns(pl.Series("gnn", lg))
    if args.split == "train":
        keep = np.isin(G.fold_e, [int(x) for x in args.score_folds.split(",")])
        out = out.filter(pl.Series(keep))
    os.makedirs(GDIR, exist_ok=True)
    out.write_parquet(os.path.join(GDIR, f"{args.out}_{args.split}_scores.parquet"))
    rep = {"args": vars(args), "feats": feats, "pairs": out.height, "runtime_s": time.time() - t0}
    print(json.dumps(rep), flush=True)
    dump(rep, "gnn", f"{args.out}_{args.split}_score.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--feats", default="")
    ap.add_argument("--s1", default="oof_xgb_c2_blend_v2.parquet", help="stage-1 table under feats/ aligned with the parts")
    ap.add_argument("--extra", default="")
    ap.add_argument("--drop_feats", default="")
    ap.add_argument("--fit_country", default="")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_s1", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--override", default="", help="cosine variant name (p2/cosvar/c2_test_<name>.parquet) or a path")
    ap.add_argument("--halves", default="A,B")
    ap.add_argument("--score_folds", default="0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.feats:
        args.feats = "c2_train_sim19" if (args.cmd == "train" or args.split == "train") else "c2_test"
    if not args.s1.startswith("feats/") and not os.path.isabs(args.s1):
        args.s1 = os.path.join("feats", args.s1)
    os.makedirs(GDIR, exist_ok=True)
    dev = device()
    {"train": cmd_train, "score": cmd_score}[args.cmd](args, dev)


if __name__ == "__main__":
    main()
