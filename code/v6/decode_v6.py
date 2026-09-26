"""V6 decoding: isotonic calibration -> best S1 per record -> [per-country EM prior correction] -> [per-source caps] ->
expected-F0.5 per S1 (optionally a different gamma for some countries).

Per-source caps: in the training ground truth no S1 has more than 5 Source-2 records (24,154 S1 have exactly 5, none
has 6) or more than 6 Source-3 records (2,816 have 6, none has 7) - results/eda/eda_report.json,
gt.per_s1_per_src_hist. The generator never links a 6th S2 or a 7th S3 record to an S1, so when the assignment gives an
S1 more than that, the least probable ones cannot all be right and are dropped before decoding.
"""
import os

import numpy as np
import polars as pl

from common.decode import assign, expected_f_decode
from common.io import CACHE, NORM
from v6.prior_em import em_prior

CAPS = (5, 6)          # max S2 / S3 records per S1 in the training ground truth


def record_source(split):
    """0 = Source 2, 1 = Source 3 for every record row (records are S2 rows then S3 rows, as ids() concatenates)."""
    n = [pl.scan_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s{k}.parquet")).select(pl.len()).collect().item()
         for k in (2, 3)]
    return np.r_[np.zeros(n[0], np.int8), np.ones(n[1], np.int8)]


def apply_caps(A, src, caps=CAPS):
    """A: one row per record (a, b, p). Keep at most caps[0] S2 and caps[1] S3 records per S1 (the most probable)."""
    A = A.with_columns(pl.Series("_s", src[A["b"].to_numpy()]))
    A = A.with_columns(pl.col("p").rank("ordinal", descending=True).over(["a", "_s"]).alias("_r"))
    lim = pl.when(pl.col("_s") == 0).then(pl.lit(caps[0])).otherwise(pl.lit(caps[1]))
    return A.filter(pl.col("_r") <= lim).drop(["_s", "_r"])


def em_correct(A, a_country, pi_tr, countries=None):
    """Saerens EM per country on the assigned probabilities (code/v6/prior_em.py)."""
    c = a_country[A["a"].to_numpy()]
    p = A["p"].to_numpy().copy()
    pis = {}
    for k in sorted(set(c.tolist())):
        if countries is not None and k not in countries:
            continue
        m = c == k
        pis[k], p[m] = em_prior(p[m], pi_tr)
    return A.with_columns(pl.Series("p", p.astype(np.float32))), pis


def decode_v6(T, iso, gamma, extra, dev, a_country, src=None, em_pi=None, gamma_by_country=None, M=4096):
    """T: a, b, p (raw score). Returns kept (a, b), the assigned table (after calibration / EM / caps) and info."""
    T = T.select(["a", "b", "p"])
    if iso is not None:
        T = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    A = assign(T, floor=0.01)
    info = {"assigned": A.height}
    if em_pi is not None:
        A, info["em_priors"] = em_correct(A, a_country, em_pi)
        info["em_pi_train"] = em_pi
    if src is not None:
        n0 = A.height
        A = apply_caps(A, src)
        info["capped_records"] = n0 - A.height
    gmap = {c: gamma for c in set(a_country.tolist())}
    gmap.update(gamma_by_country or {})
    info["gamma_by_country"] = {c: float(g) for c, g in sorted(gmap.items())}
    ac = a_country[A["a"].to_numpy()]
    keeps = []
    for g in sorted(set(gmap.values())):
        sub = A.filter(pl.Series(np.isin(ac, [c for c, v in gmap.items() if v == g])))
        if sub.height:
            keeps.append(expected_f_decode(sub, gamma=g, extra=extra, M=M, dev=str(dev))[0])
    keep = pl.concat(keeps) if keeps else A.select(["a", "b"]).head(0)
    return keep, A, info
