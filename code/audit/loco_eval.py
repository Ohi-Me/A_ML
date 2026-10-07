"""Open-set check (France has no labels): models fitted on India only, scored on US as if US were an unseen country.

Everything a France decision depends on is taken from India only: models, isotonic calibration (India folds 1-2),
rule and its settings (India folds 1-2). Reported on US fold 0 (the "unseen" country) and India fold 0 (seen).
"cal drift" = the same scores calibrated on half of US fold 0 and reported on the other half: how much is lost
because the calibration came from another country.

  python code/audit/loco_eval.py --systems s1:oof_loco_in_noemb.parquet,gnn:oof_loco_in_gnn.parquet
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.decide import Scorer, best_per_record, decide_records, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", required=True, help="name:oof_table,... under data/cache/feats")
    ap.add_argument("--fit_country", default="India")
    ap.add_argument("--out", default="loco_eval")
    args = ap.parse_args()
    gpu_init()
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    s1 = ids("train")[0]
    cty = s1["country"].to_numpy()
    half = np.array([zlib.crc32((x + "|cal").encode()) % 2 for x in s1["id"].to_list()], np.int8)
    fitc = cty == args.fit_country
    tune = np.isin(sc.fold, [1, 2]) & fitc
    evs = {c: (sc.fold == 0) & (cty == c) for c in sorted(set(cty.tolist()))}
    rep = {}
    for spec in args.systems.split(","):
        name, table = spec.split(":")
        with Timer(name):
            D = pl.read_parquet(os.path.join(CACHE, "feats", table)).select(["a", "b", "fold", "y", "p"])
            r = {}
            best = best_per_record(D)
            grid = []
            for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
                for mg in (0.2, 0.3, 0.4, 0.5, 0.6):
                    grid.append((sc.metrics(decide_records(best, thr, mg), tune)[0]["macro_f05"], thr, mg))
            f, thr, mg = max(grid)
            keep = decide_records(best, thr, mg)
            r["threshold"] = {"thr": thr, "margin": mg, "tune": f,
                              **{c: sc.metrics(keep, m)[0] for c, m in evs.items()}}
            trm = tune[D["a"].to_numpy()] & np.isin(D["fold"].to_numpy(), [1, 2])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(D["p"].to_numpy()[trm], D["y"].to_numpy()[trm])
            Dc = D.select(["a", "b"]).with_columns(pl.Series("p", iso.predict(D["p"].to_numpy()).astype(np.float32)))
            A = assign(Dc, floor=0.01)
            g = []
            for gamma in (0.8, 1.0, 1.25, 1.5):
                k, _ = expected_f_decode(A, gamma=gamma, M=1024)
                g.append((sc.metrics(k, tune)[0]["macro_f05"], gamma))
            fe, gamma = max(g)
            keep_e, _ = expected_f_decode(A, gamma=gamma, M=4096)
            r["expected_f"] = {"gamma": gamma, "tune": fe, **{c: sc.metrics(keep_e, m)[0] for c, m in evs.items()}}
            # calibration drift on the unseen country: calibrate on its own cal half, compare on its eval half
            for c in evs:
                if c == args.fit_country:
                    continue
                ev = evs[c] & (half == 1)
                cm = (evs[c] & (half == 0))[D["a"].to_numpy()]
                iso2 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(D["p"].to_numpy()[cm], D["y"].to_numpy()[cm])
                p = Dc["p"].to_numpy().copy()
                own = (cty == c)[D["a"].to_numpy()]
                p[own] = iso2.predict(D["p"].to_numpy()[own])
                k2, _ = expected_f_decode(assign(Dc.with_columns(pl.Series("p", p)), floor=0.01), gamma=gamma, M=4096)
                r[f"cal_drift_{c}"] = {"foreign_cal_eval_half": sc.metrics(keep_e, ev)[0]["macro_f05"],
                                       "own_cal_eval_half": sc.metrics(k2, ev)[0]["macro_f05"],
                                       "threshold_eval_half": sc.metrics(keep, ev)[0]["macro_f05"]}
            rep[name] = r
            print(name, json.dumps({k: (v if not isinstance(v, dict) else {c: (x["macro_f05"] if isinstance(x, dict) else x)
                                                                          for c, x in v.items()}) for k, v in r.items()}), flush=True)
    os.makedirs(os.path.join(RESULTS, "audit"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "audit", f"{args.out}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
