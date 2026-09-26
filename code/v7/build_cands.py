"""V7 candidate generation: raise the candidate ceiling (V6 on DMS: perfect scorer 0.99635, pair recall 0.988).

1. Retrieval lists (20 deep, already in the cache) re-cut on the S1 PRESENT in the universe: on the density-matched
   training universe this gives exactly the lists a blocking run on that universe's own index would give (V6 used C2
   filtered to the universe, i.e. lists cut against the full, twice as dense index: fewer and different competitors
   than test has). Depths = C2 depths (emb 5, comb 3, name 2, addr 2) x depth_mult.
2. Exact block passes on normalised fields, within country, blocks of <= cap present S1:
     x_key (name key), x_addr (address), x_numst (first house number + street core)
3. Sibling expansion (the generator is hierarchical: records of one entity share the source version's address):
     x_sib_addr  the top-1 list candidates of every other record with the identical non-empty normalised address
     x_sib_key   ... with the identical name key and state
Pass flags and list ranks (99 = not in that list) become features in pair_features.py (it keeps every column).
The train build (--universe dms) enforces the pair budget by dropping the least efficient passes and records the
passes it kept; the test build (--passes_from) uses exactly those passes.

  python code/v7/build_cands.py --split train --universe dms --name c3dms
  python code/v7/build_cands.py --split test --name c3 --passes_from c3dms
Outputs: cands/<name>_<split>.parquet (a, b, r_*, x_*), results/v7/cands_<name>_<split>.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.gpu import Timer  # noqa: E402
from common.io import CACHE, NORM, RESULTS, load_gt  # noqa: E402
from v7.keys import numstreet_key  # noqa: E402

LISTS = {"train": [("emb/e3", "emb", 5), ("blocking/b1", "comb", 3), ("blocking/b1", "name", 2), ("blocking/b1", "addr", 2)],
         "test": [("emb/e2f", "emb", 5), ("blocking/b1", "comb", 3), ("blocking/b1", "name", 2), ("blocking/b1", "addr", 2)]}
FLAGS = {"key": "x_key", "addr": "x_addr", "numstreet": "x_numst", "sib_addr": "x_sib_addr", "sib_key": "x_sib_key"}


def recut(L, present, k, rows, col):
    """(a, b, rank among present S1) for the first k present entries of each row's list."""
    out = []
    for s in range(0, len(rows), 1_000_000):
        r = rows[s:s + 1_000_000]
        M = np.asarray(L[r])
        ok = (M >= 0) & present[np.maximum(M, 0)]
        pos = np.cumsum(ok, 1) - 1
        ii, jj = np.nonzero(ok & (pos < k))
        out.append(pl.DataFrame({"a": M[ii, jj].astype(np.int32), "b": r[ii].astype(np.int32),
                                 col: pos[ii, jj].astype(np.int8)}))
    return pl.concat(out) if out else pl.DataFrame(schema={"a": pl.Int32, "b": pl.Int32, col: pl.Int8})


def block_pass(k1, kr, c1, cr, present, kept, cap, flag):
    S = pl.DataFrame({"c": c1, "k": k1, "a": np.arange(len(k1), dtype=np.int32)}).filter(pl.Series(present) & (pl.col("k") != ""))
    sz = S.group_by(["c", "k"]).len().filter(pl.col("len") <= cap)
    S = S.join(sz.select(["c", "k"]), on=["c", "k"])
    R = pl.DataFrame({"c": cr, "k": kr, "b": np.arange(len(kr), dtype=np.int32)}).filter(pl.Series(kept) & (pl.col("k") != ""))
    return R.join(S, on=["c", "k"]).select(["a", "b"]).with_columns(pl.lit(1, pl.Int8).alias(flag))


def sib_pass(gkey, kept, top1, cap, flag):
    G = pl.DataFrame({"g": gkey, "b": np.arange(len(gkey), dtype=np.int32)}).filter(pl.Series(kept) & (pl.col("g") != ""))
    sz = G.group_by("g").len().filter((pl.col("len") >= 2) & (pl.col("len") <= cap))
    G = G.join(sz.select("g"), on="g")
    J = G.join(G.rename({"b": "b2"}), on="g").filter(pl.col("b") != pl.col("b2")).select(["b", "b2"])
    return J.join(top1.rename({"b": "b2"}), on="b2").select(["a", "b"]).unique().with_columns(pl.lit(1, pl.Int8).alias(flag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--universe", default="none", choices=["none", "dms", "sim19"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--plan", default=os.path.join(RESULTS, "v7", "cands_plan.json"))
    ap.add_argument("--passes_from", default="", help="use the passes (and depth) kept by this train build")
    args = ap.parse_args()
    t0 = time.time()
    plan = json.load(open(args.plan))
    budget = float(plan.get("budget_pairs_per_record", 12.0))
    passes, mult = dict(plan["passes"]), int(plan["depth_mult"])
    if args.passes_from:
        kept_from = json.load(open(os.path.join(RESULTS, "v7", f"cands_{args.passes_from}_train.json")))
        passes, mult = kept_from["passes_kept"], int(kept_from["depth_mult"])
    cols = ["id", "country", "key", "am", "first_num", "state"]
    rd = lambda k: pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s{k}.parquet"), columns=cols)  # noqa: E731
    s1 = rd(1)
    rr = pl.concat([rd(2), rd(3)])
    n1, nr = s1.height, rr.height
    true_a = None
    if args.split == "train":
        pairs, _ = load_gt()
        T = pairs.join(pl.DataFrame({"s1": s1["id"], "a": np.arange(n1)}), on="s1") \
                 .join(pl.DataFrame({"rid": rr["id"], "b": np.arange(nr)}), on="rid")
        true_a = np.full(nr, -1, np.int64)
        true_a[T["b"].to_numpy()] = T["a"].to_numpy()
    if args.universe != "none":
        from v7.miss_probe import universe
        present, kept = universe(args.universe, s1, rr, true_a)
    else:
        present, kept = np.ones(n1, bool), np.ones(nr, bool)
    rows = np.where(kept)[0]
    rank_cols, frames, top1 = [], [], []
    with Timer(f"re-cut lists on the present S1 (depth x{mult})"):
        for d, l, k in LISTS[args.split]:
            path = os.path.join(CACHE, f"{d}_{args.split}", f"rid_{l}_idx.npy")
            if not os.path.exists(path):
                print("WARNING: missing list", path, flush=True)
                continue
            L = np.load(path, mmap_mode="r")
            col = f"r_{os.path.basename(d)}_{l}"
            rank_cols.append(col)
            frames.append(recut(L, present, min(k * mult, L.shape[1]), rows, col))
            top1.append(frames[-1].filter(pl.col(col) == 0).select(["a", "b"]))
    top1 = pl.concat(top1).unique()
    c1, cr = s1["country"].to_numpy(), rr["country"].to_numpy()
    pf = {}
    with Timer("block and sibling passes"):
        keys = {"key": (s1["key"].fill_null("").to_numpy(), rr["key"].fill_null("").to_numpy()),
                "addr": (s1["am"].fill_null("").to_numpy(), rr["am"].fill_null("").to_numpy()),
                "numstreet": (numstreet_key(s1["am"], s1["first_num"]), numstreet_key(rr["am"], rr["first_num"]))}
        for name, (a_k, b_k) in keys.items():
            if passes.get(name, 0):
                pf[name] = block_pass(a_k, b_k, c1, cr, present, kept, int(passes[name]), FLAGS[name])
        g = rr.select(
            pl.when(pl.col("am").fill_null("") != "").then(pl.concat_str(["country", "am"], separator="|")).otherwise(pl.lit("")).alias("ga"),
            pl.when(pl.col("key").fill_null("") != "").then(pl.concat_str([pl.col("country"), pl.col("key"), pl.col("state").fill_null("")],
                                                                          separator="|")).otherwise(pl.lit("")).alias("gk"))
        for name, col in (("sib_addr", "ga"), ("sib_key", "gk")):
            if passes.get(name, 0):
                pf[name] = sib_pass(g[col].to_numpy(), kept, top1, int(passes[name]), FLAGS[name])
        for name, f in pf.items():
            print(f"  pass {name}: {f.height} pairs", flush=True)
    with Timer("union (in record chunks)"):
        flags = [FLAGS[n] for n in pf]
        srcs = frames + list(pf.values())
        parts = []
        for s in range(0, nr, 2_000_000):
            sub = [f.filter((pl.col("b") >= s) & (pl.col("b") < s + 2_000_000)) for f in srcs]
            allp = pl.concat(sub, how="diagonal")
            g = allp.group_by(["a", "b"]).agg([pl.col(c).min() for c in rank_cols] + [pl.col(f).max() for f in flags])
            parts.append(g.with_columns([pl.col(c).fill_null(99).cast(pl.Int8) for c in rank_cols] +
                                        [pl.col(f).fill_null(0).cast(pl.Int8) for f in flags]))
        del srcs
        C = pl.concat(parts)
        del parts
        C = C.filter(pl.Series(present[C["a"].to_numpy()] & kept[C["b"].to_numpy()]))
    in_list = pl.any_horizontal([pl.col(c) < 99 for c in rank_cols]) if rank_cols else pl.lit(False)
    rep = {"split": args.split, "universe": args.universe, "depth_mult": mult, "passes_planned": passes,
           "records": int(kept.sum()), "list_pairs": int(C.filter(in_list).height)}
    kept_passes = dict(passes)
    if not args.passes_from:
        # enforce the pair budget: drop the least efficient passes first
        eff = plan.get("efficiency", {})
        order = sorted(pf, key=lambda n: eff.get(n, 0.0))
        for n in order:
            if C.height / max(kept.sum(), 1) <= budget:
                break
            f = FLAGS[n]
            others = [x for x in flags if x != f]
            only = ~in_list & (pl.col(f) == 1) & (pl.all_horizontal([pl.col(x) == 0 for x in others]) if others else pl.lit(True))
            before = C.height
            C = C.filter(~only).drop(f)
            flags = others
            kept_passes[n] = 0
            print(f"  budget: dropped pass {n} ({before - C.height} pairs)", flush=True)
    rep["passes_kept"] = {n: (int(v) if FLAGS[n] in flags else 0) for n, v in kept_passes.items()}
    rep["pairs"] = int(C.height)
    rep["pairs_per_record"] = C.height / max(int(kept.sum()), 1)
    rep["only_from_pass"] = {f: int(C.filter(~in_list & (pl.col(f) == 1)).height) for f in flags}
    C = C.sort(["b", "a"])
    if true_a is not None:
        m = (true_a >= 0) & kept
        m[m] = present[true_a[m]]
        Tt = pl.DataFrame({"a": true_a[m].astype(np.int32), "b": np.where(m)[0].astype(np.int32)})
        J = Tt.join(C, on=["a", "b"], how="left")
        cov = J[rank_cols[0]].is_not_null() if rank_cols else pl.Series([False] * J.height)
        rep["recall"] = float(cov.mean())
        rep["recall_lists_only"] = float(J.select(in_list.fill_null(False)).to_series().mean())
        rep["true_pairs"] = int(Tt.height)
    os.makedirs(os.path.join(CACHE, "cands"), exist_ok=True)
    C.write_parquet(os.path.join(CACHE, "cands", f"{args.name}_{args.split}.parquet"))
    rep["runtime_s"] = time.time() - t0
    os.makedirs(os.path.join(RESULTS, "v7"), exist_ok=True)
    json.dump(rep, open(os.path.join(RESULTS, "v7", f"cands_{args.name}_{args.split}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1), flush=True)
    if rep["pairs_per_record"] > budget * 1.15:
        print(f"WARNING: {rep['pairs_per_record']:.2f} pairs per record, budget {budget}", flush=True)


if __name__ == "__main__":
    main()
