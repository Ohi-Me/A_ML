"""V7 diagnostic 2: where does V6 lose F0.5 inside the candidates, and what do France decisions look like?

On DMS fold 0 (clean held-out), V6 decoded exactly as on test (isotonic folds 1-2, caps, its gamma):
  loss decomposition  1 - F0.5 summed per S1, split by the S1's error types: blocking miss only / scorer FN only /
                      FP only / mixed; and by true-match bucket (singleton, 1, 2-3, 4-6, 7+) and country
  FN kinds            best S1 of the record is another S1 (confusion) / true S1 is best but not kept (rejected) /
                      below the 0.01 floor
  samples             results/v7/fn_sample.tsv (400), fp_sample.tsv (300): raw texts of S1, record, the competing
                      S1, probabilities, empty-address / key-equal flags
France (test, no labels): results/v7/france_sample.tsv = 300 random France S1 with every candidate record, calibrated
p and the decision; label-free stats per country (predicted per S1, share of assigned records in the uncertain band,
best-p histogram) in results/v7/error_dump.json.

  python code/v7/error_dump.py
"""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.io import CACHE, NORM, RESULTS, load_source  # noqa: E402
from phase2.p2lib import device, iso_from_oof  # noqa: E402
from v6.decode_v6 import decode_v6, record_source  # noqa: E402
from v6.prior_em import assigned  # noqa: E402

TAG, CODE = "xgb_v6_c2", 62


def main():
    dev = device()
    rep = {}
    D = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{TAG}.parquet"))
    b = json.load(open(os.path.join(RESULTS, "xgboost", TAG, "decode_eval.json")))["expected_f"]["best"]
    iso = iso_from_oof(D)
    cty = ids("train")[0]["country"].to_numpy()
    sc = Scorer("train", np.load(os.path.join(CACHE, f"drop_{CODE}.npy")))
    keep, A, _ = decode_v6(D, iso, b["gamma"], b["extra"], dev, cty, src=record_source("train"))
    f, n_pred, tp = sc.per_entity(keep)
    v = sc.fold == 0
    # true pairs of fold-0 S1, in / out of candidates, kept or not
    T = sc.T.filter(pl.Series(v[sc.T["a"].to_numpy()]))
    T = T.join(D.select(["a", "b", "p"]).with_columns(pl.lit(True).alias("cand")), on=["a", "b"], how="left") \
         .join(keep.with_columns(pl.lit(True).alias("kept")), on=["a", "b"], how="left") \
         .with_columns(pl.col("cand").fill_null(False), pl.col("kept").fill_null(False))
    bm = np.bincount(T.filter(~pl.col("cand"))["a"].to_numpy(), minlength=sc.n1)       # blocking misses per S1
    fnm = np.bincount(T.filter(pl.col("cand") & ~pl.col("kept"))["a"].to_numpy(), minlength=sc.n1)
    fp = n_pred - tp
    loss = 1 - f
    kinds = {"blocking_only": (bm > 0) & (fnm == 0) & (fp == 0), "scorer_fn_only": (bm == 0) & (fnm > 0) & (fp == 0),
             "fp_only": (bm == 0) & (fnm == 0) & (fp > 0), "mixed": ((bm > 0).astype(int) + (fnm > 0) + (fp > 0)) >= 2}
    tot = float(loss[v].sum())
    rep["val_f05_fold0"] = float(f[v].mean())
    rep["loss_share_by_error_type"] = {k: float(loss[v & m].sum() / max(tot, 1e-9)) for k, m in kinds.items()}
    rep["f05_if_fixed"] = {k: float(1 - (tot - loss[v & m].sum()) / v.sum()) for k, m in kinds.items()}
    nt = sc.n_true
    buckets = {"singleton": nt == 0, "1": nt == 1, "2-3": (nt >= 2) & (nt <= 3), "4-6": (nt >= 4) & (nt <= 6), "7+": nt >= 7}
    rep["loss_share_by_true_count"] = {k: float(loss[v & m].sum() / max(tot, 1e-9)) for k, m in buckets.items()}
    rep["loss_share_by_country"] = {c: float(loss[v & (cty == c)].sum() / max(tot, 1e-9)) for c in sorted(set(cty[v].tolist()))}
    # FN kinds (inside candidates)
    A0 = assigned(D.select(["a", "b", "p"]), iso)          # best S1 per record before the caps
    best = A0.select(["b", pl.col("a").alias("best_a"), pl.col("p").alias("best_p")])
    capped = A0.join(A.select(["a", "b"]), on=["a", "b"], how="anti").select(["b"]).with_columns(pl.lit(True).alias("capped"))
    FN = T.filter(pl.col("cand") & ~pl.col("kept")).join(best, on="b", how="left").join(capped, on="b", how="left")
    FN = FN.with_columns(pl.Series("p_cal", iso.predict(FN["p"].to_numpy()).astype(np.float32)))
    ba = FN["best_a"].fill_null(-1).to_numpy()
    kind = np.where(ba < 0, "below_floor", np.where(ba != FN["a"].to_numpy(), "confused_with_other_s1",
                    np.where(FN["capped"].fill_null(False).to_numpy(), "removed_by_caps", "rejected_by_decoder")))
    FN = FN.with_columns(pl.Series("kind", kind))
    rep["fn_in_candidates"] = int(FN.height)
    rep["fn_kinds"] = {k: float((kind == k).mean()) for k in sorted(set(kind.tolist()))}
    FP = keep.filter(pl.Series(v[keep["a"].to_numpy()])).join(A, on=["a", "b"], how="left")
    FP = FP.with_columns(pl.Series("true_a", sc.true_s1_of_b[FP["b"].to_numpy()])).filter(pl.col("true_a") != pl.col("a"))
    rep["fp"] = int(FP.height)
    rep["fp_record_has_true_s1"] = float((FP["true_a"] >= 0).mean()) if FP.height else 0.0
    # texts
    s1r = load_source("train", 1)
    rr = pl.concat([load_source("train", 2), load_source("train", 3)])
    n1, na, rn, ra = (s1r["business_name"].to_numpy(), s1r["business_address"].to_numpy(),
                      rr["business_name"].to_numpy(), rr["business_address"].to_numpy())
    key1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=["key"])["key"].to_numpy()
    keyr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=["key"]) for k in (2, 3)])["key"].to_numpy()
    rng = np.random.default_rng(0)

    def txt(X, acol, ocol, n, name):
        if X.height == 0:
            return
        X = X[np.sort(rng.choice(X.height, min(n, X.height), replace=False))]
        aa, bb = X[acol].to_numpy(), X["b"].to_numpy()
        oo = X[ocol].fill_null(-1).to_numpy()
        o = np.maximum(oo, 0)
        out = X.with_columns(pl.Series("country", cty[aa]), pl.Series("s1_name", n1[aa]), pl.Series("s1_addr", na[aa]),
                             pl.Series("rec_name", rn[bb]), pl.Series("rec_addr", ra[bb]),
                             pl.Series("other_s1_name", np.where(oo >= 0, n1[o], "")),
                             pl.Series("other_s1_addr", np.where(oo >= 0, na[o], "")),
                             pl.Series("rec_addr_empty", ra[bb] == ""), pl.Series("key_eq_s1", key1[aa] == keyr[bb]))
        out.write_csv(os.path.join(RESULTS, "v7", name), separator="\t")
    os.makedirs(os.path.join(RESULTS, "v7"), exist_ok=True)
    txt(FN.select(["a", "b", "p_cal", "best_a", "best_p", "kind"]), "a", "best_a", 400, "fn_sample.tsv")
    txt(FP.select(["a", "b", "p", "true_a"]), "a", "true_a", 300, "fp_sample.tsv")
    rep["fn_empty_record_address"] = float((ra[FN["b"].to_numpy()] == "").mean()) if FN.height else 0.0
    # ---- France on test (label-free)
    tpath = os.path.join(CACHE, "feats", "test_scores_final_v6.parquet")
    if os.path.exists(tpath):
        Tt = pl.read_parquet(tpath, columns=["a", "b", "p"])
        cte = ids("test")[0]["country"].to_numpy()
        kt, At, _ = decode_v6(Tt, iso, b["gamma"], b["extra"], dev, cte, src=record_source("test"))
        pa = At["p"].to_numpy()
        ca = cte[At["a"].to_numpy()]
        npred = np.bincount(kt["a"].to_numpy(), minlength=len(cte))
        rep["test_label_free"] = {c: {"pred_per_s1": float(npred[cte == c].mean()), "empty_rate": float((npred[cte == c] == 0).mean()),
                                      "assigned_uncertain_0.05_0.95": float(((pa >= 0.05) & (pa <= 0.95))[ca == c].mean()),
                                      "assigned_p_hist": np.histogram(pa[ca == c], bins=[0, .01, .05, .2, .5, .8, .95, .99, 1.0001])[0].tolist()}
                                  for c in sorted(set(cte.tolist()))}
        fr = np.where(cte == "France")[0]
        if len(fr):
            pick = rng.choice(fr, min(300, len(fr)), replace=False)
            s1t = load_source("test", 1)
            rrt = pl.concat([load_source("test", 2), load_source("test", 3)])
            S = Tt.filter(pl.col("a").is_in(pick.tolist()))
            S = S.with_columns(pl.Series("p_cal", iso.predict(S["p"].to_numpy()).astype(np.float32)))
            S = S.join(kt.with_columns(pl.lit(True).alias("accepted")), on=["a", "b"], how="left").with_columns(pl.col("accepted").fill_null(False))
            aa, bb = S["a"].to_numpy(), S["b"].to_numpy()
            S.with_columns(pl.Series("s1_id", s1t["entity_id"].to_numpy()[aa]), pl.Series("s1_name", s1t["business_name"].to_numpy()[aa]),
                           pl.Series("s1_addr", s1t["business_address"].to_numpy()[aa]), pl.Series("rid", rrt["entity_id"].to_numpy()[bb]),
                           pl.Series("rec_name", rrt["business_name"].to_numpy()[bb]), pl.Series("rec_addr", rrt["business_address"].to_numpy()[bb])) \
             .sort(["a", "p_cal"], descending=[False, True]).write_csv(os.path.join(RESULTS, "v7", "france_sample.tsv"), separator="\t")
    json.dump(rep, open(os.path.join(RESULTS, "v7", "error_dump.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
