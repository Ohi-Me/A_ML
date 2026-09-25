"""Shared helpers for the phase-2 (V3 -> V4) investigation. Nothing here changes V2 / V3 code or artifacts.

Conventions (same as the rest of the repo):
  * score tables live under data/cache_v2 (ER_CACHE) with columns a (S1 row), b (record row), p or a logit column;
  * out-of-fold tables have a, b, fold, y, p; calibration is fitted on folds 1-2 (as in V2 / V3);
  * clean held-out folds for chain A models: 0 and 4 (fold 3 served as the XGBoost early-stopping fold, so it is
    reported but flagged).
Set ER_ALLOW_CPU=1 to run on a machine without CUDA (smoke tests only).
"""
import json
import os
import sys
import zlib

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, best_per_record, decide_records, ids  # noqa: E402
from common.decode import assign, expected_f_decode  # noqa: E402
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402

P2 = os.path.join(RESULTS, "phase2")
EVAL_FOLDS = {"f0": 0, "f4": 4, "f3_es": 3}          # f3 was the early-stopping fold of every XGBoost stage
BAND = (0.02, 0.98)                                   # uncertain band that selects cross-encoder pairs (V3)


def device():
    if torch.cuda.is_available():
        return gpu_init()
    if os.environ.get("ER_ALLOW_CPU") == "1":
        print("CPU mode (ER_ALLOW_CPU=1): smoke test only", flush=True)
        return torch.device("cpu")
    return gpu_init()                                  # exits with a clear message


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def cpath(p):
    return p if os.path.isabs(p) else os.path.join(CACHE, p)


def exists(p):
    return os.path.exists(cpath(p))


def load_scores(path, col="p", is_logit=False):
    """a, b, p (probability) from any score table under the cache."""
    T = pl.read_parquet(cpath(path)).select(["a", "b", col])
    v = T[col].to_numpy().astype(np.float64)
    if is_logit:
        v = sig(v)
    return T.select(["a", "b"]).with_columns(pl.Series("p", v.astype(np.float32)))


def fit_iso(p, y):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)


def iso_from_oof(D, folds=(1, 2), n=8_000_000, seed=0):
    tr = D.filter(pl.col("fold").is_in(list(folds)))
    if tr.height > n:
        tr = tr.sample(n=n, seed=seed)
    return fit_iso(tr["p"].to_numpy(), tr["y"].to_numpy())


def calibrate(T, iso):
    if iso is None:
        return T
    return T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))


def decode(T, rule="ef", iso=None, gamma=1.0, extra=0.0, thr=0.7, margin=0.5, dev=None, M=2048):
    """T: a, b, p (raw score as probability). Returns kept (a, b) and the assigned table (a, b, p calibrated)."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    Tc = calibrate(T, iso)
    if rule == "thr":
        return decide_records(best_per_record(Tc), thr, margin), None
    A = assign(Tc, floor=0.01)
    keep, _ = expected_f_decode(A, gamma=gamma, extra=extra, M=M, dev=str(dev))
    return keep, A


def tune_threshold(D, sc, folds=(1, 2), thrs=(0.5, 0.6, 0.65, 0.7, 0.75, 0.8), margins=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)):
    best = best_per_record(D)
    m = np.isin(sc.fold, list(folds))
    grid = [(sc.metrics(decide_records(best, t, g), m)[0]["macro_f05"], t, g) for t in thrs for g in margins]
    f, t, g = max(grid)
    return {"thr": t, "margin": g, "f05_tune": f}


def val_report(sc, keep, country, n_boot=0, seed=0):
    """F0.5 etc. on each clean fold, overall and per country (+ bootstrap CI over S1 on f0 if n_boot)."""
    f, n_pred, tp = sc.per_entity(keep)
    out = {}
    for name, k in EVAL_FOLDS.items():
        m = sc.fold == k
        r, _ = sc.metrics(keep, m)
        r["by_country"] = {}
        for c in sorted(set(country[m].tolist())):
            mc = m & (country == c)
            rc, _ = sc.metrics(keep, mc)
            r["by_country"][c] = {x: rc[x] for x in ("macro_f05", "pair_precision", "pair_recall", "fp", "fn",
                                                     "n_pred_matches", "n_pred_empty", "singleton_acc", "n_s1")}
            r["by_country"][c]["pred_per_s1"] = rc["n_pred_matches"] / max(rc["n_s1"], 1)
        r["pred_per_s1"] = r["n_pred_matches"] / max(r["n_s1"], 1)
        if n_boot and name == "f0":
            idx = np.where(m)[0]
            rng = np.random.default_rng(seed)
            bs = [float(f[rng.choice(idx, len(idx))].mean()) for _ in range(n_boot)]
            r["ci95"] = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        out[name] = r
    clean = [out[k]["macro_f05"] for k in ("f0", "f4")]
    out["mean_clean"] = float(np.mean(clean))
    out["min_clean"] = float(np.min(clean))
    return out


def uncertain_share(A, a_country, lo=0.05, hi=0.95):
    """share of assigned records (calibrated p) inside [lo, hi], per country of the S1."""
    if A is None:
        return {}
    p = A["p"].to_numpy()
    c = a_country[A["a"].to_numpy()]
    band = (p >= lo) & (p <= hi)
    return {k: float(band[c == k].mean()) for k in sorted(set(c.tolist()))}


def test_report(keep, country, extra_pairs=None, ref_keep=None):
    """label-free decision stats on test, per country. extra_pairs: frame a, b (V3-only accepts) whose survival
    is counted; ref_keep: V2 decisions for the delta."""
    a = keep["a"].to_numpy()
    n = np.bincount(a, minlength=len(country))
    out = {"matches": int(keep.height), "empty_rate": float((n == 0).mean()), "by_country": {}}
    if ref_keep is not None:
        out["delta_vs_ref"] = int(keep.height - ref_keep.height)
        both = keep.join(ref_keep, on=["a", "b"], how="semi").height
        out["only_this"], out["only_ref"] = int(keep.height - both), int(ref_keep.height - both)
    if extra_pairs is not None:
        kept = extra_pairs.join(keep, on=["a", "b"], how="semi")
        out["extra_kept"] = int(kept.height)
        out["extra_removed"] = int(extra_pairs.height - kept.height)
    rn = np.bincount(ref_keep["a"].to_numpy(), minlength=len(country)) if ref_keep is not None else None
    for c in sorted(set(country.tolist())):
        m = country == c
        d = {"s1": int(m.sum()), "matches": int(n[m].sum()), "pred_per_s1": float(n[m].mean()),
             "empty_rate": float((n[m] == 0).mean())}
        if rn is not None:
            d["delta_vs_ref"] = int(n[m].sum() - rn[m].sum())
        if extra_pairs is not None:
            ea = extra_pairs["a"].to_numpy()
            em = m[ea]
            ex = extra_pairs.filter(pl.Series(em))
            d["extra"] = int(ex.height)
            d["extra_removed"] = int(ex.height - ex.join(keep, on=["a", "b"], how="semi").height)
        out["by_country"][c] = d
    return out


def s1_hash_fold(split):
    """fold_of(S1 id) for every S1 row of a split (test rows get a pseudo fold the same way)."""
    s1 = ids(split)[0]["id"].to_list()
    return np.array([zlib.crc32(x.encode()) % 5 for x in s1], np.int8)


def dump(obj, *path):
    p = os.path.join(P2, *path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    print("wrote", p, flush=True)
    return p


def ctx_features(C, col):
    """competition features of one cosine column exactly as pair_features.py / simulate_drop.py compute them.
    C: a, b, <col> over the whole candidate table. Returns C with b_gap/b_rank/a_gap/a_rank/b_gap2 of col."""
    C = C.with_columns([
        (pl.col(col) - pl.col(col).max().over("b")).alias(f"b_gap_{col}"),
        pl.col(col).rank("ordinal", descending=True).over("b").cast(pl.Int32).alias(f"b_rank_{col}"),
        (pl.col(col) - pl.col(col).max().over("a")).alias(f"a_gap_{col}"),
        pl.col(col).rank("ordinal", descending=True).over("a").cast(pl.Int32).alias(f"a_rank_{col}")])
    sec = C.filter(pl.col(f"b_rank_{col}") == 2).select(["b", pl.col(col).alias("_sec")])
    return C.join(sec, on="b", how="left").with_columns(
        (pl.col(col) - pl.col("_sec").fill_null(0.0)).alias(f"b_gap2_{col}")).drop("_sec")


COS_CTX = ["cos_e3", "b_gap_cos_e3", "b_rank_cos_e3", "a_gap_cos_e3", "a_rank_cos_e3", "b_gap2_cos_e3"]


def apply_override(d, ov):
    """replace columns of a feature part by the override table (a, b, cols...) joined on (a, b)."""
    if ov is None:
        return d
    cols = [c for c in ov.columns if c not in ("a", "b")]
    j = d.select(["a", "b"]).join(ov, on=["a", "b"], how="left", maintain_order="left")
    assert j.height == d.height
    miss = j[cols[0]].null_count()
    assert miss == 0, f"override misses {miss} pairs"
    return d.with_columns([j[c].cast(d.schema[c]).alias(c) if c in d.columns else j[c] for c in cols])


QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def qstats_all(x):
    """mean / std / quantiles of the finite values (+ share of missing)."""
    x = np.asarray(x, np.float64)
    fin = np.isfinite(x)
    out = {"n": int(len(x)), "missing_share": float(1 - fin.mean()) if len(x) else 0.0}
    x = x[fin]
    if len(x):
        out.update({"mean": float(x.mean()), "std": float(x.std()),
                    **{f"q{int(round(q * 100)):02d}": float(v) for q, v in zip(QS, np.quantile(x, QS))}})
    return out
