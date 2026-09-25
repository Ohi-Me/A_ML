"""Official metric: macro F0.5 over Source-1 entities, singletons included
(empty truth: 1.0 if we predict nothing, else 0.0)."""
import numpy as np


def f05_entity(pred, true, beta=0.5):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def evaluate(pred, truth, s1_ids, cand=None):
    """pred, truth, cand: dict s1 -> iterable of ids. Returns a dict of the standard metrics."""
    scores, tp, fp, fn, n_pred_match, n_empty = [], 0, 0, 0, 0, 0
    cand_hit = cand_tot = 0
    sing_ok = sing_n = 0
    for s in s1_ids:
        P, T = set(pred.get(s, ())), set(truth.get(s, ()))
        f = f05_entity(P, T)
        scores.append(f)
        tp += len(P & T)
        fp += len(P - T)
        fn += len(T - P)
        n_pred_match += len(P)
        n_empty += (len(P) == 0)
        if not T:
            sing_n += 1
            sing_ok += (len(P) == 0)
        if cand is not None:
            C = set(cand.get(s, ()))
            cand_hit += len(C & T)
            cand_tot += len(T)
    out = {
        "n_s1": len(s1_ids), "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "pair_precision": tp / max(tp + fp, 1), "pair_recall": tp / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "n_pred_matches": n_pred_match, "n_pred_empty": n_empty,
        "n_true_singletons": sing_n, "singleton_acc": sing_ok / max(sing_n, 1),
    }
    p, r = out["pair_precision"], out["pair_recall"]
    out["pair_f05"] = 1.25 * p * r / max(0.25 * p + r, 1e-12)
    if cand is not None:
        out["candidate_recall"] = cand_hit / max(cand_tot, 1)
        out["n_candidates"] = int(sum(len(set(v)) for v in cand.values()))
    return out, np.array(scores)
