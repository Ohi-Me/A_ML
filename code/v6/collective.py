"""Collective (sibling) evidence for a candidate pair (a, b): what do the S1's OTHER likely records say about b?

Why: the data comes from a hierarchical generator (V6_RESEARCH.md §1.3). Each source holds a version of the entity,
and records are noisy copies of it. Same-source siblings share address details that the S1 lacks (81 % identical
normalised address vs 76 % record-vs-S1, 60 % cross-source), and a changed house number is usually shared by several
siblings of that source. A hard record (new DBA name, different number, empty address) can therefore be linked
through an easy sibling. V2's stage-2 sibling features only took the max TF-IDF / embedding cosine to *confident*
siblings. Here the evidence is exact (hash equality of normalised fields), split by same / cross source, weighted by
the sibling's probability, and turned into a record-level margin over the record's other candidate S1.

Features (0 when the S1 has no qualifying sibling):
  c_addr_same / c_addr_cross   max p(sibling) over siblings with the identical normalised address, same / other source
  c_num_same / c_num_cross     max p(sibling) over siblings with the same first house number
  c_num_nonS1                  ... where that shared number differs from the S1's own number (source-version signature)
  c_key                        max p(sibling) over siblings with the identical name key
  c_support                    sum over siblings of p * (addr_eq + num_eq + key_eq) / 3
  c_nsib / c_nsib_same         number of confident siblings (p >= 0.5), all / same source
  c_src_conf                   confident records of this S1 from b's source (per-source caps: S2 <= 5, S3 <= 6)
  c_support_margin             c_support minus the best c_support of the record's other candidate S1
"""
import os

import numpy as np
import polars as pl

from common.io import CACHE, NORM

COLS = ["c_addr_same", "c_addr_cross", "c_num_same", "c_num_cross", "c_num_nonS1", "c_key", "c_support",
        "c_nsib", "c_nsib_same", "c_src_conf", "c_support_margin"]


def _h(s):
    """u64 hash of a string column, 0 for empty / null (so 'both empty' never counts as equal)."""
    s = s.fill_null("")
    h = s.hash(seed=0).to_numpy().astype(np.uint64)
    h[(s == "").to_numpy()] = 0
    return h


def record_attrs(split):
    """per-record arrays: source (0 = S2, 1 = S3), hashes of the normalised address, first house number and name key;
    per-S1 hash of the first house number."""
    s2 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s2.parquet"), columns=["am", "first_num", "key"])
    s3 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s3.parquet"), columns=["am", "first_num", "key"])
    rr = pl.concat([s2, s3])
    s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s1.parquet"), columns=["first_num"])
    src = np.r_[np.zeros(s2.height, np.int8), np.ones(s3.height, np.int8)]
    return {"src": src, "ah": _h(rr["am"]), "nh": _h(rr["first_num"]), "kh": _h(rr["key"]), "s1_nh": _h(s1["first_num"])}


def collective_features(D, attrs, conf_thr=0.3, chunk_a=60_000):
    """D: frame with r (row id), a, b, p. Returns D with the COLS added (p is the probability the evidence uses)."""
    src, ah, nh, kh, s1nh = attrs["src"], attrs["ah"], attrs["nh"], attrs["kh"], attrs["s1_nh"]
    conf = D.filter(pl.col("p") >= conf_thr).select(["a", pl.col("b").alias("b2"), pl.col("p").alias("p2")])
    a_max = int(D["a"].max()) + 1 if D.height else 0
    outs = []
    for s in range(0, a_max, chunk_a):
        d = D.filter((pl.col("a") >= s) & (pl.col("a") < s + chunk_a)).select(["r", "a", "b"])
        c = conf.filter((pl.col("a") >= s) & (pl.col("a") < s + chunk_a))
        j = d.join(c, on="a").filter(pl.col("b") != pl.col("b2"))
        if j.height == 0:
            continue
        b = j["b"].to_numpy().astype(np.int64)
        b2 = j["b2"].to_numpy().astype(np.int64)
        a = j["a"].to_numpy().astype(np.int64)
        p2 = j["p2"].to_numpy().astype(np.float32)
        same = src[b] == src[b2]
        aeq = (ah[b] != 0) & (ah[b] == ah[b2])
        neq = (nh[b] != 0) & (nh[b] == nh[b2])
        non1 = neq & (nh[b] != s1nh[a])
        keq = (kh[b] != 0) & (kh[b] == kh[b2])
        conf5 = p2 >= 0.5
        J = pl.DataFrame({"r": j["r"], "x_as": np.where(aeq & same, p2, 0), "x_ac": np.where(aeq & ~same, p2, 0),
                          "x_ns": np.where(neq & same, p2, 0), "x_nc": np.where(neq & ~same, p2, 0),
                          "x_n1": np.where(non1, p2, 0), "x_k": np.where(keq, p2, 0),
                          "x_sup": p2 * (aeq.astype(np.float32) + neq + keq) / 3,
                          "x_c": conf5.astype(np.int32), "x_cs": (conf5 & same).astype(np.int32)})
        outs.append(J.group_by("r").agg(
            pl.col("x_as").max().alias("c_addr_same"), pl.col("x_ac").max().alias("c_addr_cross"),
            pl.col("x_ns").max().alias("c_num_same"), pl.col("x_nc").max().alias("c_num_cross"),
            pl.col("x_n1").max().alias("c_num_nonS1"), pl.col("x_k").max().alias("c_key"),
            pl.col("x_sup").sum().alias("c_support"), pl.col("x_c").sum().alias("c_nsib"),
            pl.col("x_cs").sum().alias("c_nsib_same")))
    # b's source is attached while the rows are still in D's order (joins below may reorder rows)
    base = D.drop([c for c in COLS if c in D.columns]).with_columns(
        pl.Series("src_b", src[D["b"].to_numpy().astype(np.int64)]))
    if outs:
        S = pl.concat(outs)
        base = base.join(S, on="r", how="left")
    else:
        base = base.with_columns([pl.lit(None, pl.Float32).alias(c) for c in COLS[:9]])
    base = base.with_columns([pl.col(c).fill_null(0).cast(pl.Float32) for c in COLS[:9]])
    # confident records of this S1 per source (per-source caps), joined on b's source
    cc = base.filter(pl.col("p") >= 0.5).group_by(["a", "src_b"]).len().rename({"len": "c_src_conf"})
    base = base.join(cc, on=["a", "src_b"], how="left").with_columns(pl.col("c_src_conf").fill_null(0).cast(pl.Float32))
    # record-level collective margin over the record's other candidate S1
    top = base.group_by("b").agg(pl.col("c_support").max().alias("_t1"),
                                 pl.col("c_support").sort(descending=True).slice(1, 1).first().alias("_t2"))
    base = base.join(top, on="b", how="left").with_columns(
        (pl.col("c_support") - pl.when(pl.col("c_support") < pl.col("_t1")).then(pl.col("_t1"))
         .otherwise(pl.col("_t2").fill_null(0.0))).cast(pl.Float32).alias("c_support_margin")).drop(["_t1", "_t2", "src_b"])
    return base.sort("r")
