"""Compare decision rules on out-of-fold scores: threshold+margin vs expected-F0.5 decoding (with isotonic
calibration fitted on folds 1-2). Settings are chosen on folds 1-2 and reported on fold 0.

  python code/xgboost/decode_eval.py --tag xgb_c1_s2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record, decide_records  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402
from common.split import VAL_FOLD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sub", default="xgboost")
    ap.add_argument("--simdrop", type=int, default=0)
    args = ap.parse_args()
    gpu_init()
    t0 = time.time()
    rdir = os.path.join(RESULTS, args.sub, args.tag)
    D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.tag}.parquet"))
    sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{args.simdrop}.npy")) if args.simdrop else None)
    tune = np.isin(sc.fold, [1, 2])
    val = sc.fold == VAL_FOLD
    rep = {}
    with Timer("baseline threshold rule"):
        best = best_per_record(D)
        grid = []
        for thr in (0.5, 0.6, 0.65, 0.7, 0.75):
            for mg in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                m, _ = sc.metrics(decide_records(best, thr, mg), tune)
                grid.append((m["macro_f05"], thr, mg))
        f, thr, mg = max(grid)
        mv, _ = sc.metrics(decide_records(best, thr, mg), val)
        rep["threshold_rule"] = {"thr": thr, "margin": mg, "f05_tune": f, "val": mv}
        print("threshold rule", rep["threshold_rule"], flush=True)
    with Timer("isotonic calibration on folds 1-2"):
        tr = D.filter(pl.col("fold").is_in([1, 2])).sample(n=min(8_000_000, D.height), seed=0)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(tr["p"].to_numpy(), tr["y"].to_numpy())
        D = D.with_columns(pl.Series("p", iso.predict(D["p"].to_numpy()).astype(np.float32)))
    with Timer("expected-F decoding grid"):
        A = assign(D, floor=0.01)
        grid = []
        for gamma in (0.8, 1.0, 1.25, 1.5):
            for extra in (0.0, 0.05):
                keep, _ = expected_f_decode(A, gamma=gamma, extra=extra, M=1024)
                m, _ = sc.metrics(keep, tune)
                grid.append({"gamma": gamma, "extra": extra, "f05_tune": m["macro_f05"]})
                print(grid[-1], flush=True)
        b = max(grid, key=lambda r: r["f05_tune"])
        keep, ks = expected_f_decode(A, gamma=b["gamma"], extra=b["extra"], M=4096)
        mv, fv = sc.metrics(keep, val)
        rep["expected_f"] = {"best": b, "grid": grid, "val": mv, "breakdown": sc.breakdown(fv, val)}
        print("expected-F decode", json.dumps(rep["expected_f"]["best"]), json.dumps(mv), flush=True)
    rep["runtime_s"] = time.time() - t0
    json.dump(rep, open(os.path.join(rdir, "decode_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
