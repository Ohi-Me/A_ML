"""V6-PL: adapt the V6 stage 1 to the unseen test country (France) with conservative pseudo-labels, then replay the
V6 collective rounds with the adapted stage-1 scores. Seen countries (US, India) keep their V6 scores.

  select : V6 test scores (round 2, isotonic from the OOF table) on unseen-country pairs -> pseudo-labels
           (code/v5/pl_lib.py rule: confident best S1 with address agreement -> y=1 and y=0 for the record's other
           candidates; confident distractor records -> y=0; everything in between left out). The same rule is
           checked on India -> US by loco_eval.py (step pl) before it is trusted (code/v6/choose_v6.py).
  fit    : stage-1 full / no-embedding refit on all folds of the training universe + the pseudo-labelled test rows
           (x1.25 rounds, like the V6 refit)
  score  : unseen-country pairs -> PL stage-1 blend (w_unseen) -> V6 rounds 1 and 2 with the refit round models of
           predict_v6 (cached in results/final/final_v6/) -> feats/test_scores_final_v6pl.parquet

  python code/v6/pl_v6.py select
  python code/v6/pl_v6.py fit
  python code/v6/pl_v6.py score
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
from phase2.p2lib import device, iso_from_oof  # noqa: E402
from v5.pl_lib import AGREE_COLS, MixIter, select_pseudo  # noqa: E402
from v6.collective import record_attrs  # noqa: E402
from v6.predict_v6 import REN, add_args, collective_rounds, score_parts, stage1_blend, test_extras  # noqa: E402

RD = os.path.join(RESULTS, "v6")
PLF = os.path.join(CACHE, "v6", "pl_test.parquet")


def unseen_mask():
    seen = set(ids("train")[0]["country"].unique().to_list())
    c = ids("test")[0]["country"].to_numpy()
    return ~np.isin(c, list(seen)), c


def files(args):
    trf = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    tef = sorted(glob.glob(os.path.join(CACHE, "feats", args.test_feats, "part_*.parquet")))
    return trf, tef


def cmd_select(args):
    um, cty = unseen_mask()
    _, tef = files(args)
    T = pl.read_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.base}.parquet"), columns=["a", "b", "p"])
    T = T.filter(pl.Series(um[T["a"].to_numpy()]))
    iso = iso_from_oof(pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.c2}.parquet")))
    T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    F = pl.concat([pl.read_parquet(f, columns=["a", "b"] + AGREE_COLS) for f in tef]).join(T.select(["a", "b"]), on=["a", "b"], how="semi")
    rows, rep = select_pseudo(T, F, args.pos_thr, args.margin, args.neg_thr, args.addr_jac, args.max_pairs)
    os.makedirs(os.path.dirname(PLF), exist_ok=True)
    rows.write_parquet(PLF)
    rep.update({"unseen_countries": sorted(set(cty[um].tolist())), "args": vars(args)})
    os.makedirs(RD, exist_ok=True)
    json.dump(rep, open(os.path.join(RD, "pl_select.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


def cmd_fit(args):
    import xgboost as xgb
    bl = json.load(open(os.path.join(RESULTS, "xgboost", args.s1, "report.json")))["blend"]
    rows = pl.read_parquet(PLF)
    trf, tef = files(args)
    for tag in (bl["full"], bl["noemb"]):
        path = os.path.join(RD, f"pl_stage1_{tag}.json")
        if os.path.exists(path):
            print("exists", path)
            continue
        rp = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))
        n = max(int(rp["rounds"] * args.round_mult), 1)
        with Timer(f"PL refit {tag}: universe folds 0-4 + {rows.height} pseudo-labelled test pairs, {n} rounds"):
            it = MixIter(trf, rp["features"], [0, 1, 2, 3, 4], extra_files=tef, pl_rows=rows, rename=REN)
            d = xgb.QuantileDMatrix(it, max_bin=int(rp["params"].get("max_bin", 256)))
            m = xgb.train(rp["params"], d, n)
            m.save_model(path)
            print("rows (train + pseudo)", it.last[0], "pseudo rows", it.last[1], flush=True)
            del d, m


def cmd_score(args):
    import xgboost as xgb
    dev = device()
    t0 = time.time()
    bl = json.load(open(os.path.join(RESULTS, "xgboost", args.s1, "report.json")))["blend"]
    um, cty = unseen_mask()
    trf, tef = files(args)
    seen = set(ids("train")[0]["country"].unique().to_list())
    models = []
    for tag in (bl["full"], bl["noemb"]):
        feats = json.load(open(os.path.join(RESULTS, "xgboost", tag, "report.json")))["features"]
        models.append((xgb.Booster(model_file=os.path.join(RD, f"pl_stage1_{tag}.json")), feats))
    D1 = pl.read_parquet(os.path.join(CACHE, "feats", f"test_stage1_{args.base}.parquet"))
    rows = um[D1["a"].to_numpy()]
    with Timer(f"PL stage 1 on {int(rows.sum())} unseen-country pairs"):
        AB, (pf, pn) = score_parts(tef, models, rows=rows)
        U = stage1_blend(AB, pf, pn, bl, cty, seen).rename({"p": "p_pl"})
    D1 = D1.join(U, on=["a", "b"], how="left", maintain_order="left").with_columns(
        pl.coalesce(["p_pl", "p"]).cast(pl.Float32).alias("p")).drop("p_pl")
    D1.write_parquet(os.path.join(CACHE, "feats", f"test_stage1_{args.out}.parquet"))
    # the round models were refit by predict_v6 (--base): reuse them
    rdir = os.path.join(RESULTS, "final", args.base)
    extras = test_extras(args.c1, args.c2)
    p_r1, p_r2 = collective_rounds(D1, trf, tef, args, rdir, dev, record_attrs("test"), extras)
    base = pl.read_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.base}.parquet"))
    assert (base["b"].to_numpy() == D1["b"].to_numpy()).all()
    p = np.where(rows, p_r2, base["p"].to_numpy()).astype(np.float32)
    out = base.with_columns(pl.Series("p", p), pl.Series("p_stage1", D1["p"]),
                            pl.Series("p_round1", np.where(rows, p_r1, base["p_round1"].to_numpy()).astype(np.float32)))
    out.write_parquet(os.path.join(CACHE, "feats", f"test_scores_{args.out}.parquet"))
    rep = {"unseen_pairs": int(rows.sum()), "mean_p_before": float(base["p"].to_numpy()[rows].mean()) if rows.any() else None,
           "mean_p_after": float(p[rows].mean()) if rows.any() else None,
           "share_p_gt_0.5_before": float((base["p"].to_numpy()[rows] > 0.5).mean()) if rows.any() else None,
           "share_p_gt_0.5_after": float((p[rows] > 0.5).mean()) if rows.any() else None, "runtime_s": time.time() - t0}
    json.dump(rep, open(os.path.join(RD, "pl_score.json"), "w"), indent=1)
    print(json.dumps(rep), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "fit", "score"])
    add_args(ap)
    ap.set_defaults(out="final_v6pl")
    ap.add_argument("--base", default="final_v6", help="the V6 run whose test scores / refit round models are used")
    ap.add_argument("--pos_thr", type=float, default=0.97)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--neg_thr", type=float, default=0.02)
    ap.add_argument("--addr_jac", type=float, default=0.3)
    ap.add_argument("--max_pairs", type=int, default=4_000_000)
    args = ap.parse_args()
    os.makedirs(RD, exist_ok=True)
    {"select": cmd_select, "fit": cmd_fit, "score": cmd_score}[args.step](args)


if __name__ == "__main__":
    main()
