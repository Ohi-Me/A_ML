"""Data probe (minutes, CPU): test the structural hypotheses behind the V6 plan on the FULL training data.

  T1 row order      does the file row order carry the match? (Spearman of S1 row vs record row over true pairs; are
                    same-entity records adjacent in their file?) A strong effect would be a generator artefact.
                    Using it may breach the fair-play review, so it is only measured here, never used by a model.
  T2 source version are same-source siblings (records of one S1 from the same source) closer to each other than to
                    the S1? (identical normalised address; shared house number that differs from S1's)
  T3 empty address  how many empty-address true records have several S1 with the same name key (ambiguous), and
                    does a record-count prior (the true S1 has fewer other records) separate them?
  T4 density        S1 / records / records-per-S1 per country: train, SIM19, test (why US test inflates)

  python code/v6/data_probe.py            -> results/v6/data_probe.json
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import CACHE, NORM, RESULTS, load_gt  # noqa: E402


def fmean(s):
    v = s.mean() if len(s) else None
    return float(v) if v is not None else None


def norm(split, k, cols):
    return pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s{k}.parquet"), columns=cols)


def main():
    t0 = time.time()
    rep = {}
    s1 = norm("train", 1, ["id", "country", "key", "am", "first_num"]).with_row_index("row1")
    r2 = norm("train", 2, ["id", "country", "key", "am", "first_num"]).with_row_index("rowf").with_columns(pl.lit("S2").alias("src"))
    r3 = norm("train", 3, ["id", "country", "key", "am", "first_num"]).with_row_index("rowf").with_columns(pl.lit("S3").alias("src"))
    rr = pl.concat([r2, r3])
    pairs, _ = load_gt()
    P = pairs.join(s1.select(["id", "row1", "country", pl.col("key").alias("k1"), pl.col("am").alias("m1"),
                              pl.col("first_num").alias("f1")]), left_on="s1", right_on="id") \
             .join(rr.select(["id", "rowf", "src", pl.col("key").alias("k2"), pl.col("am").alias("m2"),
                              pl.col("first_num").alias("f2")]), left_on="rid", right_on="id")
    # ---- T1 row order
    t1 = {}
    for src in ("S2", "S3"):
        q = P.filter(pl.col("src") == src)
        rho = q.select(pl.corr("row1", "rowf", method="spearman")).item()
        g = q.sort(["s1", "rowf"]).group_by("s1", maintain_order=True).agg(pl.col("rowf").diff().drop_nulls().alias("d"))
        d = g.explode("d")["d"].drop_nulls().to_numpy()
        n = rr.filter(pl.col("src") == src).height
        t1[src] = {"spearman_s1row_vs_recordrow": float(rho), "sibling_gap_median": float(np.median(d)) if len(d) else None,
                   "sibling_adjacent_share": float(np.mean(np.abs(d) == 1)) if len(d) else None,
                   "random_expectation_adjacent": float(2 / n)}
    rep["T1_row_order"] = t1
    # ---- T2 source-level versions: pairs of records of the same S1
    sib = P.select(["s1", "rid", "src", "m1", "m2", "f1", "f2"])
    sib = sib.join(sib, on="s1", suffix="_o").filter(pl.col("rid") < pl.col("rid_o"))
    sib = sib.filter((pl.col("m2") != "") & (pl.col("m2_o") != ""))
    sib = sib.with_columns((pl.col("src") == pl.col("src_o")).alias("same_src"),
                           (pl.col("m2") == pl.col("m2_o")).alias("addr_eq"),
                           ((pl.col("f2") == pl.col("f2_o")) & (pl.col("f2") != "") & (pl.col("f2") != pl.col("f1"))).alias("shared_non_s1_num"))
    t2 = {}
    for flag in (True, False):
        q = sib.filter(pl.col("same_src") == flag)
        t2["same_source" if flag else "cross_source"] = {"pairs": q.height, "identical_normalised_address": fmean(q["addr_eq"]),
                                                          "share_same_number_that_differs_from_S1": fmean(q["shared_non_s1_num"])}
    rs = P.filter(pl.col("m2") != "")
    t2["record_vs_S1"] = {"pairs": rs.height, "identical_normalised_address": fmean(rs["m1"] == rs["m2"]),
                          "record_number_differs_from_S1": fmean((rs["f1"] != rs["f2"]) & (rs["f2"] != "") & (rs["f1"] != ""))}
    rep["T2_source_versions"] = t2
    # ---- T3 empty-address ambiguity + record-count prior
    keyn = s1.group_by(["country", "key"]).len().rename({"len": "same_key_s1"})
    cnt = P.group_by("s1").len().rename({"len": "n_records"})
    E = P.filter(pl.col("m2") == "").join(keyn, left_on=["country", "k2"], right_on=["country", "key"], how="left") \
         .with_columns(pl.col("same_key_s1").fill_null(0))
    amb = E.filter(pl.col("same_key_s1") >= 2)
    # among S1 sharing the record's key: is the true S1 the one with the fewest other true records?
    S1k = s1.select(["id", "country", "key"]).join(cnt, left_on="id", right_on="s1", how="left").with_columns(pl.col("n_records").fill_null(0))
    j = amb.select(["s1", "rid", "country", "k2"]).join(S1k, left_on=["country", "k2"], right_on=["country", "key"])
    j = j.with_columns((pl.col("id") == pl.col("s1")).alias("is_true"),
                       (pl.col("n_records") - (pl.col("id") == pl.col("s1")).cast(pl.Int64)).alias("others"))
    g = j.group_by("rid").agg(pl.col("others").filter(pl.col("is_true")).first().alias("true_others"),
                              pl.col("others").filter(~pl.col("is_true")).min().alias("min_other_others"),
                              pl.len().alias("m"))
    rep["T3_empty_address"] = {
        "empty_address_true_records": E.height,
        "with_unique_name_key": int((E["same_key_s1"] == 1).sum()),
        "with_ambiguous_name_key(>=2 S1)": amb.height,
        "ambiguous_mean_candidates": float(g["m"].mean()) if g.height else None,
        "count_prior_true_has_fewer_records": float((g["true_others"] < g["min_other_others"]).mean()) if g.height else None,
        "count_prior_true_has_more_records": float((g["true_others"] > g["min_other_others"]).mean()) if g.height else None,
        "random_guess_accuracy": float((1 / g["m"]).mean()) if g.height else None}
    # ---- T4 density
    dens = {}
    for split in ("train", "test"):
        a = norm(split, 1, ["country"]).group_by("country").len().rename({"len": "s1"})
        b = pl.concat([norm(split, k, ["country"]) for k in (2, 3)]).group_by("country").len().rename({"len": "records"})
        for c, n1, nr in a.join(b, on="country").iter_rows():
            dens.setdefault(c, {})[split] = {"s1": n1, "records": nr, "records_per_s1": nr / n1}
    dp = os.path.join(CACHE, "drop_19.npy")
    if os.path.exists(dp):
        dm = np.load(dp)
        c1 = s1["country"].to_numpy()
        for c in dens:
            if "train" in dens[c]:
                kept = int(((c1 == c) & ~dm).sum())
                dens[c]["sim19"] = {"s1": kept, "records_per_s1": dens[c]["train"]["records"] / kept}
    for c, d in dens.items():
        if "train" in d and "test" in d:
            d["test_s1_over_train_s1"] = d["test"]["s1"] / d["train"]["s1"]
    rep["T4_density"] = dens
    rep["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "v6"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v6", "data_probe.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
