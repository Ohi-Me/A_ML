"""V7 diagnostic 1: which true pairs does candidate generation miss, and what would each new blocking pass recover?

Why: on the density-matched universe (DMS) a perfect scorer on the C2 candidates reaches only F0.5 0.99635 (pair
recall 0.988; results/v6/v6_choice.json, ceiling_dms). Validation >= 0.998 is impossible until candidate recall is
~99.8 %. This script measures, on the training labels only:

  c2_as_filtered   C2 recall inside the universe (what V6 trained on)
  recut            the retrieval lists (20 deep) re-cut after removing S1 absent from the universe = what blocking
                   would give if it ran on the universe's own index (as on test, whose index is sparse); at the C2
                   depths (emb 5, comb 3, name 2, addr 2), x2 and full 20
  passes on top of recut C2 depth, each with the pairs it adds per record:
     key<=N        every S1 of the country with the record's exact normalised name key (blocks of <= N S1)
     addr<=N       every S1 with the record's exact normalised address (non-empty)
     numstreet<=N  same first house number and same street core (address tokens without numbers)
     sib_addr      top-2 recut candidates of the record's siblings = other records with the identical non-empty
                   normalised address (hierarchical generator: 81 % of same-source siblings share it)
     sib_key       same with siblings = records with the identical name key and state (blocks <= 30)
  miss profile     empty address, key / address equality, script, source, rank in each list
Writes results/v7/miss_probe.json and results/v7/missed_sample.tsv (500 missed pairs with the raw texts).

  python code/v7/miss_probe.py [--universe dms|sim19|full]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.io import CACHE, NORM, RESULTS, load_gt, load_source  # noqa: E402
from v6.simulate_universe import h  # noqa: E402
from v7.keys import numstreet_key  # noqa: E402

LISTS = [("emb/e3", "emb", 5), ("blocking/b1", "comb", 3), ("blocking/b1", "name", 2), ("blocking/b1", "addr", 2)]
PLANS = {"dms": {"US": (0.62, 0.19), "India": (1.0, 0.19)}, "sim19": {"US": (1.0, 0.19), "India": (1.0, 0.19)},
         "full": {}}


def universe(kind, s1, rr, true_a):
    plan = PLANS[kind]
    c1, cr = s1["country"].to_numpy(), rr["country"].to_numpy()
    keep1 = np.array([plan.get(c, (1.0, 0.0))[0] for c in c1])
    drop1 = np.array([plan.get(c, (1.0, 0.0))[1] for c in c1])
    kept_s1 = h(s1["id"].to_list(), "|keep") < keep1
    present = kept_s1 & ~(h(s1["id"].to_list(), "|drop") < drop1)
    keepr = np.array([plan.get(c, (1.0, 0.0))[0] for c in cr])
    kept_r = np.where(true_a >= 0, kept_s1[np.maximum(true_a, 0)], h(rr["id"].to_list(), "|keep") < keepr)
    return present, kept_r


def present_rank(L, a, present):
    """L: (n, K) list rows, a: (n,) target S1. Rank of a among the present entries of each row (99 = not listed)."""
    ok = (L >= 0) & present[np.maximum(L, 0)]
    pos = np.cumsum(ok, 1) - 1
    hit = ok & (L == a[:, None])
    return np.where(hit.any(1), np.where(hit, pos, 99).min(1), 99)


def recut_sets(L, present, k):
    """per row, the first k present entries of L (-1 padded)."""
    ok = (L >= 0) & present[np.maximum(L, 0)]
    pos = np.cumsum(ok, 1) - 1
    keep = ok & (pos < k)
    return np.where(keep, L, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="dms", choices=list(PLANS))
    ap.add_argument("--sample_records", type=int, default=300_000, help="records used to estimate pairs per record")
    args = ap.parse_args()
    t0 = time.time()
    want = ["id", "country", "key", "am", "first_num", "state", "nonlatin"]
    def rd(k):
        path = os.path.join(CACHE, f"norm_{NORM}_train_s{k}.parquet")
        have = pl.read_parquet_schema(path)
        d = pl.read_parquet(path, columns=[c for c in want if c in have])
        return d if "nonlatin" in d.columns else d.with_columns(pl.lit(False).alias("nonlatin"))
    s1 = rd(1)
    parts = [rd(2), rd(3)]
    rr = pl.concat(parts)
    src = np.r_[np.zeros(parts[0].height, np.int8), np.ones(parts[1].height, np.int8)]
    pairs, _ = load_gt()
    T = pairs.join(pl.DataFrame({"s1": s1["id"], "a": np.arange(s1.height)}), on="s1") \
             .join(pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height)}), on="rid").select(["a", "b"])
    true_a = np.full(rr.height, -1, np.int64)
    true_a[T["b"].to_numpy()] = T["a"].to_numpy()
    present, kept_r = universe(args.universe, s1, rr, true_a)
    T = T.filter(pl.Series(present[T["a"].to_numpy()] & kept_r[T["b"].to_numpy()]))
    a, b = T["a"].to_numpy().astype(np.int64), T["b"].to_numpy().astype(np.int64)
    rep = {"universe": args.universe, "true_pairs": int(len(a)), "present_s1": int(present.sum()),
           "kept_records": int(kept_r.sum()), "lists": {}}
    # ---- C2 as filtered into the universe
    C = pl.read_parquet(os.path.join(CACHE, "cands", "c2_train.parquet"), columns=["a", "b"])
    inC = T.join(C.with_columns(pl.lit(True).alias("h")), on=["a", "b"], how="left")["h"].fill_null(False).to_numpy()
    rep["recall_c2_as_filtered"] = float(inC.mean())
    # ---- list ranks among present S1 (recut)
    Ls, ranks = {}, {}
    for d, l, k in LISTS:
        path = os.path.join(CACHE, f"{d}_train", f"rid_{l}_idx.npy")
        if not os.path.exists(path):
            print("missing list", path, flush=True)
            continue
        Ls[l] = (np.load(path, mmap_mode="r"), k)
        ranks[l] = present_rank(np.asarray(Ls[l][0][b]), a, present)
        rep["lists"][l] = {"depth": int(Ls[l][0].shape[1]), "c2_depth": k,
                           **{f"recall_top{kk}": float((ranks[l] < kk).mean()) for kk in (1, 2, 3, 5, 10, 20)}}
    def cover(mult=1.0, full=False):
        c = np.zeros(len(a), bool)
        for l, (L, k) in Ls.items():
            kk = L.shape[1] if full else max(1, int(round(k * mult)))
            c |= ranks[l] < kk
        return c
    base = cover()
    rep["recall_recut"] = {"c2_depths": float(base.mean()), "x2_depths": float(cover(2).mean()),
                           "x4_depths": float(cover(4).mean()), "all_20": float(cover(full=True).mean())}
    # pairs per record of the recut union (sample of kept records)
    rng = np.random.default_rng(0)
    kb = np.where(kept_r)[0]
    sb = np.sort(rng.choice(kb, min(args.sample_records, len(kb)), replace=False))
    ppr = {}
    for name, mult, full in (("c2_depths", 1, False), ("x2_depths", 2, False), ("x4_depths", 4, False), ("all_20", 1, True)):
        cs = [recut_sets(np.asarray(L[sb]), present, L.shape[1] if full else max(1, int(round(k * mult)))) for L, k in Ls.values()]
        M = np.concatenate(cs, 1)
        M = np.sort(M, 1)
        uniq = (M >= 0) & np.r_["1", np.ones((len(M), 1), bool), M[:, 1:] != M[:, :-1]]
        ppr[name] = float(uniq.sum(1).mean())
    rep["pairs_per_record_recut"] = ppr
    # ---- extra passes on top of recut C2 depths
    cty1, ctyr = s1["country"].to_numpy(), rr["country"].to_numpy()
    def block_pass(key_s1, key_r, caps):
        """S1 sharing an exact key with the record (within country), blocks of <= cap present S1."""
        S = pl.DataFrame({"c": cty1, "k": key_s1, "p": present}).filter(pl.col("p") & (pl.col("k") != ""))
        sizes = S.group_by(["c", "k"]).len()
        R = pl.DataFrame({"c": ctyr, "k": key_r}).with_row_index("b")
        Rs = R.join(sizes, on=["c", "k"], how="left").sort("b")
        size_r = Rs["len"].fill_null(0).to_numpy()
        same = (key_s1[a] == key_r[b]) & (key_r[b] != "")
        out = {}
        for cap in caps:
            ok = same & (size_r[b] <= cap)
            m = kept_r & (size_r <= cap)
            out[str(cap)] = {"recall_gain": float((ok & ~base).mean()), "pairs_per_record": float(size_r[m].sum() / kept_r.sum())}
        return out, same
    k1, kr = s1["key"].fill_null("").to_numpy(), rr["key"].fill_null("").to_numpy()
    m1, mr = s1["am"].fill_null("").to_numpy(), rr["am"].fill_null("").to_numpy()
    n1, nr = s1["first_num"].fill_null("").to_numpy(), rr["first_num"].fill_null("").to_numpy()
    ns1, nsr = numstreet_key(s1["am"], s1["first_num"]), numstreet_key(rr["am"], rr["first_num"])
    rep["passes"] = {}
    rep["passes"]["key"], key_eq = block_pass(k1, kr, (1, 3, 5, 10, 30))
    rep["passes"]["addr"], addr_eq = block_pass(m1, mr, (1, 3, 5, 10))
    rep["passes"]["numstreet"], ns_eq = block_pass(ns1, nsr, (1, 3, 5, 10))
    # sibling expansion: siblings' recut top-2 per list
    miss = np.where(~base)[0]
    def sib_pass(gkey, cap):
        G = pl.DataFrame({"g": gkey, "kr": kept_r}).with_row_index("b").filter(pl.col("kr") & (pl.col("g") != ""))
        sz = G.group_by("g").len().filter(pl.col("len") <= cap)
        G = G.join(sz.select("g"), on="g")
        Mi = pl.DataFrame({"i": miss, "b": b[miss], "a": a[miss]}).join(G.select(["b", "g"]), on="b")
        J = Mi.join(G.select([pl.col("b").alias("b2"), "g"]), on="g").filter(pl.col("b2") != pl.col("b"))
        if J.height == 0:
            return {"recall_gain": 0.0, "misses_with_sibling": 0.0}
        hit = np.zeros(J.height, bool)
        b2, aa = J["b2"].to_numpy(), J["a"].to_numpy()
        for L, k in Ls.values():
            hit |= present_rank(np.asarray(L[b2]), aa, present) < 2
        got = J.with_columns(pl.Series("h", hit)).group_by("i").agg(pl.col("h").any())
        return {"recall_gain": float(got["h"].sum() / len(a)),
                "misses_with_sibling": float(Mi["i"].n_unique() / max(len(miss), 1))}
    gk = rr.select(
        pl.when(pl.col("am").fill_null("") != "").then(pl.concat_str(["country", "am"], separator="|")).otherwise(pl.lit("")).alias("ga"),
        pl.when(pl.col("key").fill_null("") != "").then(pl.concat_str([pl.col("country"), pl.col("key"), pl.col("state").fill_null("")],
                                                                      separator="|")).otherwise(pl.lit("")).alias("gk"))
    rep["passes"]["sib_addr"] = sib_pass(gk["ga"].to_numpy(), 50)
    rep["passes"]["sib_key"] = sib_pass(gk["gk"].to_numpy(), 30)
    allp = base | (key_eq & True) | addr_eq | ns_eq
    rep["recall_recut_plus_all_exact_passes_uncapped"] = float(allp.mean())
    # ---- profile of the misses (recut C2 depths)
    emp = mr[b] == ""
    prof = {"n_missed": int(len(miss)), "share_of_true_pairs": float(len(miss) / len(a)),
            "empty_record_address": float(emp[miss].mean()), "key_equal": float(key_eq[miss].mean()),
            "addr_equal": float(addr_eq[miss].mean()), "numstreet_equal": float(ns_eq[miss].mean()),
            "first_num_equal": float(((n1[a] == nr[b]) & (nr[b] != ""))[miss].mean()),
            "record_nonlatin": float(rr["nonlatin"].fill_null(False).to_numpy().astype(bool)[b][miss].mean()),
            "source_S3": float(src[b][miss].mean()),
            "by_country": {c: float((ctyr[b][miss] == c).mean()) for c in sorted(set(ctyr.tolist()))},
            "rank_in_lists": {l: {"1-5": float(((r[miss] >= 0) & (r[miss] < 5)).mean()),
                                  "5-20": float(((r[miss] >= 5) & (r[miss] < 20)).mean()),
                                  "absent": float((r[miss] >= 20).mean())} for l, r in ranks.items()}}
    rep["miss_profile"] = prof
    # ---- readable sample
    s1r = load_source("train", 1)
    rrr = pl.concat([load_source("train", 2), load_source("train", 3)])
    smp = rng.choice(miss, min(500, len(miss)), replace=False) if len(miss) else miss
    top = {l: np.asarray(L[b[smp]])[:, 0] for l, (L, _) in Ls.items()}
    tb = next(iter(top.values())) if top else np.full(len(smp), -1)
    D = pl.DataFrame({"s1_id": s1r["entity_id"].to_numpy()[a[smp]], "rid": rrr["entity_id"].to_numpy()[b[smp]],
                      "s1_name": s1r["business_name"].to_numpy()[a[smp]], "rec_name": rrr["business_name"].to_numpy()[b[smp]],
                      "s1_addr": s1r["business_address"].to_numpy()[a[smp]], "rec_addr": rrr["business_address"].to_numpy()[b[smp]],
                      "country": ctyr[b[smp]], "key_eq": key_eq[smp], "addr_eq": addr_eq[smp],
                      **{f"rank_{l}": ranks[l][smp] for l in ranks},
                      "top1_emb_name": np.where(tb >= 0, s1r["business_name"].to_numpy()[np.maximum(tb, 0)], ""),
                      "top1_emb_addr": np.where(tb >= 0, s1r["business_address"].to_numpy()[np.maximum(tb, 0)], "")})
    os.makedirs(os.path.join(RESULTS, "v7"), exist_ok=True)
    D.write_csv(os.path.join(RESULTS, "v7", "missed_sample.tsv"), separator="\t")
    rep["runtime_s"] = time.time() - t0
    json.dump(rep, open(os.path.join(RESULTS, "v7", "miss_probe.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)


if __name__ == "__main__":
    main()
