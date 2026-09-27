"""V8 diagnostic: is there sibling structure in the NAMES that could resolve the hardest records?

The hardest block of true pairs: records with an empty address whose name key is shared by many S1 (147 k true
pairs, 82 S1 per key on average; results/v6/data_probe.json T3). Text similarity to the S1 can't pick the right one.
The generator is hierarchical (V6 data study: same-source siblings share the source version's address 48 % of the
time, vs 36 % record-vs-S1). If they also share the source version's NAME spelling (legal words, typos, word order),
an empty-address record can be linked through a sibling that does have an address.

Measured on the density-matched universe (DMS), training labels only:
  per true record: does another record of the SAME S1 have the identical normalised full name `n` (same source /
  other source), or the identical raw name (lower case, spaces collapsed)? Does such a sibling have an address?
  purity: among all records with that name in the country, the share that belong to the record's S1
  group sizes of (country, n)
  split: all true records / empty-address true records / empty-address with an ambiguous key (>= 2 present S1)
ID check (report only, not used by any model): rank correlation between the numeric part of S1 ids and record ids over
true pairs, and the same for random pairs. A leak would show as |spearman| >> 0.

  python code/v8/sib_probe.py   -> results/v8/sib_probe.json
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import CACHE, NORM, RESULTS, load_gt, load_source  # noqa: E402
from v7.keys import name_sib_key  # noqa: E402
from v7.miss_probe import universe  # noqa: E402


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main():
    t0 = time.time()
    cols = ["id", "country", "key", "n", "am"]
    rd = lambda k: pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet"), columns=cols)  # noqa: E731
    s1 = rd(1)
    p2, p3 = rd(2), rd(3)
    rr = pl.concat([p2, p3])
    src = np.r_[np.zeros(p2.height, np.int8), np.ones(p3.height, np.int8)]
    raw = pl.concat([load_source("train", 2), load_source("train", 3)])["business_name"].fill_null("")
    raw_key = (rr["country"] + "|" + raw.str.to_lowercase().str.replace_all(r"\s+", " ").str.strip_chars()).to_numpy()
    raw_key[raw.str.strip_chars().to_numpy() == ""] = ""
    pairs, _ = load_gt()
    T = pairs.join(pl.DataFrame({"s1": s1["id"], "a": np.arange(s1.height)}), on="s1") \
             .join(pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height)}), on="rid").select(["a", "b"])
    true_a = np.full(rr.height, -1, np.int64)
    true_a[T["b"].to_numpy()] = T["a"].to_numpy()
    present, kept = universe("dms", s1, rr, true_a)
    rep = {"universe": "dms"}
    nkey = name_sib_key(rr["country"], rr["n"])
    am_empty = (rr["am"].fill_null("") == "").to_numpy()
    # present S1 per (country, key): is the record's key ambiguous?
    ks = s1.filter(pl.Series(present)).group_by(["country", "key"]).len().rename({"len": "n_s1_key"})
    amb = rr.select(["country", "key"]).with_row_index("b").join(ks, on=["country", "key"], how="left").sort("b")[
        "n_s1_key"].fill_null(0).to_numpy() >= 2
    # records in the universe with their true S1 (-1 = distractor)
    ta = np.where(kept & (true_a >= 0), true_a, -1)
    ta[ta >= 0] = np.where(present[ta[ta >= 0]], ta[ta >= 0], -1)
    R = pl.DataFrame({"b": np.arange(rr.height), "a": ta, "src": src, "g": nkey, "gr": raw_key, "emp": am_empty,
                      "amb": amb}).filter(pl.Series(kept))

    def stats(gcol, mask_name, mask):
        G = R.filter(pl.col(gcol) != "")
        sz = G.group_by(gcol).len().rename({"len": "gsize"})
        # per (group, S1): records of that S1 in the group, same source split
        per = G.filter(pl.col("a") >= 0).group_by([gcol, "a"]).agg(pl.len().alias("k"), pl.col("src").n_unique().alias("nsrc"))
        X = R.filter(pl.Series(mask[R["b"].to_numpy()]) & (pl.col("a") >= 0) & (pl.col(gcol) != "")) \
             .join(per, on=[gcol, "a"], how="left").join(sz, on=gcol, how="left")
        # siblings = other records of the same S1 with the same name; same source = another record of that S1 in the
        # group from the record's source
        same_src = G.filter(pl.col("a") >= 0).group_by([gcol, "a", "src"]).agg(pl.len().alias("ks"),
                                                                                (~pl.col("emp")).sum().alias("ks_addr"))
        X = X.join(same_src, on=[gcol, "a", "src"], how="left")
        addr_sib = G.filter((pl.col("a") >= 0) & ~pl.col("emp")).group_by([gcol, "a"]).len().rename({"len": "k_addr"})
        X = X.join(addr_sib, on=[gcol, "a"], how="left")
        k = X["k"].fill_null(1).to_numpy()
        ks_ = X["ks"].fill_null(1).to_numpy()
        kad = X["k_addr"].fill_null(0).to_numpy() - (~X["emp"].to_numpy()).astype(int)
        gs = X["gsize"].fill_null(1).to_numpy()
        n = max(X.height, 1)
        return {"records": int(X.height),
                "has_sibling_same_name": float((k >= 2).mean()),
                "has_same_source_sibling_same_name": float((ks_ >= 2).mean()),
                "has_sibling_same_name_with_address": float((kad >= 1).mean()),
                "purity_mean": float(((k - 1) / np.maximum(gs - 1, 1))[k >= 2].mean()) if (k >= 2).any() else 0.0,
                "group_size_le_5": float((gs <= 5).sum() / n), "group_size_le_20": float((gs <= 20).sum() / n),
                "sibling_and_group_le_10": float(((k >= 2) & (gs <= 10)).mean())}
    for gcol, gname in (("g", "normalised_name"), ("gr", "raw_name")):
        rep[gname] = {"all_true": stats(gcol, "all", np.ones(rr.height, bool)),
                      "empty_address": stats(gcol, "empty", am_empty),
                      "empty_address_ambiguous_key": stats(gcol, "empty_amb", am_empty & amb)}
    # ---- ID check (report only)
    def num(ids):
        return pl.Series(ids).str.extract(r"(\d+)$").cast(pl.Int64, strict=False).fill_null(-1).to_numpy()
    n1, nr = num(s1["id"]), num(rr["id"])
    a, b = T["a"].to_numpy(), T["b"].to_numpy()
    rng = np.random.default_rng(0)
    i = rng.choice(len(a), min(2_000_000, len(a)), replace=False)
    rb = rng.choice(rr.height, len(i))
    rep["id_check"] = {"spearman_true_pairs": spearman(n1[a[i]], nr[b[i]]),
                       "spearman_random_pairs": spearman(n1[a[i]], nr[rb]),
                       "median_abs_diff_true": float(np.median(np.abs(n1[a[i]] - nr[b[i]]))),
                       "median_abs_diff_random": float(np.median(np.abs(n1[a[i]] - nr[rb]))),
                       "note": "report only; no model reads ids"}
    rep["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "v8"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v8", "sib_probe.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
