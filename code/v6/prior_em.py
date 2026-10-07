"""Label-shift (prior) correction per country without labels: Saerens, Latinne & Decaestecker (2002) EM on calibrated
probabilities (Alexandari et al., ICML 2020: EM + bias-corrected calibration is hard to beat).

Population: each record's best S1 after calibration (the pairs the decoder can accept). Training prior pi_tr = mean
calibrated p of that population on the training universe. On a new population (test country) EM estimates its prior
pi and rescales every probability: p' = r p / (r p + s (1 - p)), r = pi / pi_tr, s = (1 - pi) / (1 - pi_tr).
It is only used on test if it helps on a labelled shifted universe (code/v6/score_chain.py reports it).
"""
import numpy as np
import polars as pl

from common.decode import assign, expected_f_decode


def em_prior(p, pi_tr, iters=200, tol=1e-7):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    pi = float(pi_tr)
    q = p
    for _ in range(iters):
        r, s = pi / pi_tr, (1 - pi) / (1 - pi_tr)
        q = r * p / (r * p + s * (1 - p))
        new = float(q.mean())
        if abs(new - pi) < tol:
            pi = new
            break
        pi = new
    return pi, q.astype(np.float32)


def assigned(T, iso):
    """calibrated best-S1-per-record table (a, b, p) from raw scores T (a, b, p)."""
    Tc = T.with_columns(pl.Series("p", iso.predict(T["p"].to_numpy()).astype(np.float32)))
    return assign(Tc, floor=0.01)


def train_prior(oof, iso):
    """pi_tr from the training universe's OOF table (all folds; calibrated, best S1 per record)."""
    return float(assigned(oof.select(["a", "b", "p"]), iso)["p"].mean())


def em_decode(T, iso, a_country, pi_tr, gamma=1.0, extra=0.0, dev=None, M=4096, countries=None):
    """decode with per-country EM-corrected probabilities. countries: which countries to correct (None = all).
    Returns kept (a, b), the corrected assigned table and the estimated priors."""
    A = assigned(T, iso)
    c = a_country[A["a"].to_numpy()]
    p = A["p"].to_numpy().copy()
    pis = {}
    for k in sorted(set(c.tolist())):
        if countries is not None and k not in countries:
            continue
        m = c == k
        pis[k], p[m] = em_prior(p[m], pi_tr)
    A = A.with_columns(pl.Series("p", p))
    keep, _ = expected_f_decode(A, gamma=gamma, extra=extra, M=M, dev=dev)
    return keep, A, pis
