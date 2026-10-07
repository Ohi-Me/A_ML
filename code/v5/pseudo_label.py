"""V5-PL: adapt stage 1 to the unseen country(ies) of the test set with conservative pseudo-labels.

  select : V5 test scores (stage 2, isotonic from the OOF table) on unseen-country pairs -> pseudo-labels
           (code/v5/pl_lib.py rule, validated on India -> US by loco_eval.py step pl)
  fit    : stage-1 full / no-embedding refit on all training folds + the pseudo-labelled test rows (x1.25 rounds,
           same as the V2/V5 refit)
  score  : unseen-country pairs -> PL stage-1 blend -> stage-2 context + sibling features -> V5 stage-2 refit model;
           writes feats/test_scores_final_v5pl.parquet = V5 test scores with the unseen-country rows replaced.
Seen countries (US, India) keep the V5 scores unchanged.

  python code/v5/pseudo_label.py select
  python code/v5/pseudo_label.py fit
  python code/v5/pseudo_label.py score
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
from v5.pl_lib import AGREE_COLS, MixIter, select_pseudo  # noqa: E402

TR, TE = "c2v5_train_sim19", "c2v5_test"
BLEND, S2 = "xgb_v5_blend", "xgb_v5_s2"
RD = os.path.join(RESULTS, "v5")
PLF = os.path.join(CACHE, "v5", "pl_test.parquet")


def unseen_mask():
    seen = set(ids("train")[0]["country"].unique().to_list())
    c = ids("test")[0]["country"].to_numpy()
    return ~np.isin(c, list(seen)), c


def te_files():
    return sorted(glob.glob(os.path.join(CACHE, "feats", TE, "part_*.parquet")))


def cmd_select(args):
    from phase2.p2lib import iso_from_oof
    um, cty = unseen_mask()
    T = pl.read_parquet(os.path.join(CACHE, "feats", "test_scores_final_v5.parquet")).select(["a", "b", "p"])
    T = T.filter(pl.Series(um[T["a"].to_numpy()]))
    iso = iso_from_oof(pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{S2}.parquet")))
    T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    F = pl.concat([pl.read_parquet(f, columns=["a", "b"] + AGREE_COLS) for f in te_files()]).join(T.select(["a", "b"]), on=["a", "b"], how="semi")
    rows, rep = select_pseudo(T, F, args.pos_thr, args.margin, args.neg_thr, args.addr_jac, args.max_pairs)
    os.makedirs(os.path.dirname(PLF), exist_ok=True)
    rows.write_parquet(PLF)
    rep.update({"unseen_countries": sorted(set(cty[um].tolist())), "args": vars(args)})
    os.makedirs(RD, exist_ok=True)
    json.dump(rep, open(os.path.join(RD, "pl_select.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


def cmd_fit(args):
    import xgboost as xgb
    bl = json.load(open(os.path.join(RESULTS, "xgboost", BLEND, "report.json")))["blend"]
    rows = pl.read_parquet(PLF)
    trf = sorted(glob.glob(os.path.join(CACHE, "feats", TR, "part_*.parquet")))
    for tag in (bl["full"], bl["noemb"]):
        rp = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
        n = int(rp["rounds"] * args.round_mult)
        with Timer(f"PL refit {tag}: train folds 0-4 + {rows.height} pseudo-labelled test pairs, {n} rounds"):
            it = MixIter(trf, rp["features"], [0, 1, 2, 3, 4], extra_files=te_files(), pl_rows=rows, rename={"r_e2f_emb": "r_e3_emb"})
            d = xgb.QuantileDMatrix(it, max_bin=int(rp["params"].get("max_bin", 256)))
            m = xgb.train(rp["params"], d, n)
            m.save_model(os.path.join(RD, f"pl_stage1_{tag}.json"))
            print("rows (train + pseudo)", it.last[0], "pseudo rows", it.last[1], flush=True)
            del d, m


def cmd_score(args):
    import scipy.sparse as sp
    import torch
    import xgboost as xgb
    from common.stage2_feats import context_features, sibling_features
    from phase2.p2lib import device
    dev = device()
    t0 = time.time()
    bl = json.load(open(os.path.join(RESULTS, "xgboost", BLEND, "report.json")))["blend"]
    um, cty = unseen_mask()
    parts = []
    for f in te_files():
        d = pl.read_parquet(f).rename({"r_e2f_emb": "r_e3_emb"}, strict=False)
        parts.append(d.filter(pl.Series(um[d["a"].to_numpy()])))
    Dfull = pl.concat(parts)
    del parts
    ps = []
    for tag in (bl["full"], bl["noemb"]):
        feats = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))["features"]
        m = xgb.Booster(model_file=os.path.join(RD, f"pl_stage1_{tag}.json"))
        ps.append(m.predict(xgb.DMatrix(Dfull.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy())))
    p1 = (bl["w_unseen"] * ps[0] + (1 - bl["w_unseen"]) * ps[1]).astype(np.float32)
    D = Dfull.select(["a", "b"]).with_columns(pl.Series("p", p1)).with_row_index("r")
    with Timer("stage-2 features on the unseen-country pairs"):
        D = context_features(D)
        E = torch.from_numpy(np.load(os.path.join(CACHE, "emb", "e2f_test", "emb.npy"))).to(dev)
        bdir = os.path.join(CACHE, "blocking", "b1_test")
        NAME = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        ADDR = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        D = sibling_features(D, E, NAME, ADDR, ids("test")[0].height, dev).sort("r")
        del E, NAME, ADDR
    new_cols = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
    X2 = pl.concat([Dfull, D.select(new_cols).rename({"p": "s1_p"})], how="horizontal")
    f2 = json.load(open(os.path.join(RESULTS, "xgboost", S2, "report.json")))["features"]
    m2 = xgb.Booster(model_file=os.path.join(RESULTS, "final", "final_v5", "stage2_all_folds.json"))
    p2 = m2.predict(xgb.DMatrix(X2.select([pl.col(c).cast(pl.Float32) for c in f2]).to_numpy())).astype(np.float32)
    U = X2.select(["a", "b"]).with_columns(pl.Series("p_pl", p2))
    base = pl.read_parquet(os.path.join(CACHE, "feats", "test_scores_final_v5.parquet"))
    out = base.join(U, on=["a", "b"], how="left").with_columns(pl.coalesce(["p_pl", "p"]).alias("p")).drop("p_pl")
    out.write_parquet(os.path.join(CACHE, "feats", "test_scores_final_v5pl.parquet"))
    chg = base.join(U, on=["a", "b"])
    rep = {"unseen_pairs": U.height, "mean_p_before": float(chg["p"].mean()), "mean_p_after": float(chg["p_pl"].mean()),
           "share_p_gt_0.5_before": float((chg["p"] > 0.5).mean()), "share_p_gt_0.5_after": float((chg["p_pl"] > 0.5).mean()),
           "runtime_s": time.time() - t0}
    json.dump(rep, open(os.path.join(RD, "pl_score.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "fit", "score"])
    ap.add_argument("--pos_thr", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--neg_thr", type=float, default=0.02)
    ap.add_argument("--addr_jac", type=float, default=0.3)
    ap.add_argument("--max_pairs", type=int, default=4_000_000)
    ap.add_argument("--round_mult", type=float, default=1.25)
    args = ap.parse_args()
    os.makedirs(RD, exist_ok=True)
    {"select": cmd_select, "fit": cmd_fit, "score": cmd_score}[args.step](args)


if __name__ == "__main__":
    main()
