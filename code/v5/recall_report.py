"""Where are the remaining errors? Pipeline metrics per stage, per country, on the clean held-out folds (0 and 4).

  candidate recall          true pairs present in the candidate table
  stage-1 recall@0.5        true pairs with stage-1 p >= 0.5
  top-1 recall              true pairs that are their record's best S1 under the final scorer
  final                     precision / recall / FP / FN / no-match (singleton) accuracy / macro F0.5 (iso + EF decode)
  FN by cause               not a candidate | record assigned to another S1 | best but rejected by the decoder
  FP by cause               record is a distractor (no true S1) | record belongs to another S1  (+ FP on true singletons)
  blocking headroom         missing true pairs whose S1 and record share the exact name key, or the same first house
                            number + a shared street-core token (what exact / field-wise blocking would add)

  python code/v5/recall_report.py --s1 xgb_v5_blend --s2 xgb_v5_s2
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.decide import Scorer, ids  # noqa: E402
from common.io import CACHE, NORM, RESULTS  # noqa: E402
from phase2.p2lib import decode, device, iso_from_oof  # noqa: E402
from v5.extra_feats import STREET_WORDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", default="xgb_v5_blend")
    ap.add_argument("--s2", default="xgb_v5_s2")
    ap.add_argument("--dec", default="")
    args = ap.parse_args()
    dev = device()
    t0 = time.time()
    sc = Scorer("train", np.load(os.path.join(CACHE, "drop_19.npy")))
    cty = sc.country
    S1 = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.s1}.parquet"), columns=["a", "b", "p"])
    S2 = pl.read_parquet(os.path.join(CACHE, "feats", f"oof_{args.s2}.parquet"))
    dj = json.load(open(args.dec or os.path.join(RESULTS, "xgboost", args.s2, "decode_eval.json")))["expected_f"]["best"]
    keep, _ = decode(S2.select(["a", "b", "p"]), rule="ef", iso=iso_from_oof(S2), gamma=dj["gamma"], extra=dj["extra"], dev=dev, M=4096)
    T = sc.T                                                     # true pairs of kept (non-dropped) S1
    cand = S2.select(["a", "b"]).with_columns(pl.lit(True).alias("is_cand"))
    best = S2.sort(["b", "p"], descending=[False, True]).group_by("b", maintain_order=True).agg(pl.col("a").first().alias("a_best"))
    Tx = T.join(cand, on=["a", "b"], how="left").join(best, on="b", how="left") \
          .join(S1.rename({"p": "p1"}), on=["a", "b"], how="left") \
          .join(keep.with_columns(pl.lit(True).alias("acc")), on=["a", "b"], how="left") \
          .with_columns(pl.col("is_cand").fill_null(False), pl.col("acc").fill_null(False))
    ta = Tx["a"].to_numpy()
    K = keep.with_columns(pl.Series("true_a", sc.true_s1_of_b[keep["b"].to_numpy()]))
    ka = K["a"].to_numpy()
    # blocking headroom: missing true pairs sharing the exact key, or first number + a street-core token
    s1n = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=["key", "first_num", "am"])
    rrn = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=["key", "first_num", "am"]) for k in (2, 3)])
    miss = Tx.filter(~pl.col("is_cand")).select(["a", "b"])
    words = list(STREET_WORDS)
    stc = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(  # noqa: E731
        pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d") & ~pl.element().is_in(words)))
    M = miss.with_columns(pl.Series("k1", s1n["key"].gather(miss["a"])), pl.Series("k2", rrn["key"].gather(miss["b"])),
                          pl.Series("f1", s1n["first_num"].gather(miss["a"])), pl.Series("f2", rrn["first_num"].gather(miss["b"])),
                          pl.Series("m1", s1n["am"].gather(miss["a"])), pl.Series("m2", rrn["am"].gather(miss["b"])),
                          pl.Series("e2", (rrn["am"].gather(miss["b"]).fill_null("") == "")))
    M = M.with_columns(stc("m1").alias("t1"), stc("m2").alias("t2"))
    M = M.with_columns(((pl.col("k1") == pl.col("k2")) & (pl.col("k1") != "")).alias("key_eq"),
                       ((pl.col("f1") == pl.col("f2")) & (pl.col("f1") != "") &
                        (pl.col("t1").list.set_intersection(pl.col("t2")).list.len() > 0)).alias("num_street"))
    mcty = cty[M["a"].to_numpy()]
    rep = {"s1": args.s1, "s2": args.s2, "decoder": dj, "by_split": {}}
    for fname, fk in (("f0", 0), ("f4", 4)):
        for c in ["ALL"] + sorted(set(cty.tolist())):
            mask = (sc.fold == fk) & ((cty == c) if c != "ALL" else True)
            tm = mask[ta]
            t = Tx.filter(pl.Series(tm))
            if t.height == 0:
                continue
            m, _ = sc.metrics(keep, mask)
            kk = K.filter(pl.Series(mask[ka]))
            fp = kk.filter(pl.col("true_a") != pl.col("a"))
            fn = t.filter(~pl.col("acc"))
            mm = M.filter(pl.Series((sc.fold[M["a"].to_numpy()] == fk) & ((mcty == c) if c != "ALL" else True)))
            rep["by_split"][f"{fname}|{c}"] = {
                "n_s1": int(mask.sum()), "true_pairs": t.height,
                "candidate_recall": float(t["is_cand"].mean()),
                "stage1_recall_at_0.5": float((t["p1"].fill_null(0) >= 0.5).mean()),
                "top1_recall": float((t["a_best"] == t["a"]).fill_null(False).mean()),
                "final": {k: m[k] for k in ("macro_f05", "pair_precision", "pair_recall", "fp", "fn", "singleton_acc", "n_pred_empty")},
                "fn_by_cause": {"not_a_candidate": int((~fn["is_cand"]).sum()),
                                "record_assigned_to_other_s1": int((fn["is_cand"] & (fn["a_best"] != fn["a"])).sum()),
                                "best_but_rejected": int((fn["is_cand"] & (fn["a_best"] == fn["a"])).sum())},
                "fp_by_cause": {"record_is_distractor": int((fp["true_a"] < 0).sum()),
                                "record_of_other_s1": int((fp["true_a"] >= 0).sum()),
                                "fp_on_true_singleton_s1": int((sc.n_true[fp["a"].to_numpy()] == 0).sum())},
                "blocking_headroom": {"missing_pairs": mm.height, "record_address_empty": int(mm["e2"].sum()),
                                      "exact_key_equal": int(mm["key_eq"].sum()), "same_number_and_street_token": int(mm["num_street"].sum()),
                                      "either": int((mm["key_eq"] | mm["num_street"]).sum())}}
    rep["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "v5"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v5", f"recall_{args.s2}.json"), "w"), indent=1)
    for k, v in rep["by_split"].items():
        if k.startswith("f0"):
            print(k, json.dumps({x: v[x] for x in ("candidate_recall", "top1_recall")}), v["final"]["macro_f05"], v["fn_by_cause"], v["fp_by_cause"],
                  v["blocking_headroom"], flush=True)


if __name__ == "__main__":
    main()
