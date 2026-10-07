"""V5 extra pair features: country-agnostic evidence meant to transfer to an unseen country (France).

Why: V2/V3 learnt on US/India, where business names are distinctive. In France (unseen, 15 % of test S1) names are
built from a small generic vocabulary ("Nantes Club SAS", "Association de Formation") in a few dense cities, so a
high name similarity is weak evidence there and the address must decide. The models need to *see* how ambiguous a
name is and how specific the shared tokens are, instead of learning country-specific name statistics.

New columns (prefix v5_), appended to a copy of the feature parts (so every existing script works on the new table):
  record-side ambiguity from the TF-IDF name / address retrieval lists (dropped SIM19 S1 are removed):
    v5_{name,addr}_top    best retrieval score of the record
    v5_{name,addr}_n05    number of S1 within 0.05 of that best score (how many look equally good)
    v5_{name,addr}_n15    ... within 0.15
    v5_{name,addr}_gap12  best minus second best
    v5_{name,addr}_sc     this S1's score in the list (0 if not listed), v5_{name,addr}_gap = top - sc
  name specificity (document frequency of the name tokens among the country's S1 names, log1p):
    v5_s1_mindf, v5_s1_meandf, v5_rec_mindf, v5_shared_mindf (rarest shared token; -1 if none), v5_n_shared_rare
  street core (address tokens without numbers, street types and noise words such as the 'ndeg' left by 'N°'):
    v5_st_jac, v5_st_c12, v5_st_c21, v5_st_tset

  python code/v5/extra_feats.py --src c2_train_sim19          -> feats/c2v5_train_sim19/
  python code/v5/extra_feats.py --src c2_test [--cosvar hash1] -> feats/c2v5_test/
V6 (any simulated universe): the S1 absent from the universe (drop_<code>.npy) leave the retrieval lists AND the name
document-frequency table, so every count is what it would be if the universe were the real S1 file (as on test):
  python code/v5/extra_feats.py --src c2_train_dmsA --absent_code 62 --out c2v6_train_dmsA
  python code/v5/extra_feats.py --src c2_test --out c2v6_test
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common.normalize as N  # noqa: E402
from common.io import CACHE, NORM, RESULTS  # noqa: E402

STREET_WORDS = set(N.ADDR_MAP.values()) | set(N.ADDR_MAP.keys()) | N.ADDR_NOISE | {
    "ndeg", "deg", "n", "no", "bis", "ter", "b", "a", "c", "d", "de", "du", "des", "la", "le", "les", "l", "et", "of",
    "the", "and"}
RARE = 5


def load_norm(split):
    s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s1.parquet"), columns=["country", "core", "am"])
    rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{split}_s{k}.parquet"), columns=["country", "core", "am"])
                    for k in (2, 3)])
    return s1, rr


def token_lists(df):
    words = list(STREET_WORDS)
    return df.select(
        pl.col("core").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("nt"),
        pl.col("am").fill_null("").str.split(" ").list.eval(
            pl.element().filter((pl.element() != "") & ~pl.element().str.contains(r"\d") & ~pl.element().is_in(words))
        ).list.unique().alias("st"),
        pl.col("country"))


def df_table(s1tok, present=None):
    """(country, tok) -> number of S1 whose name contains tok (only S1 present in the universe, if given)."""
    if present is not None:
        s1tok = s1tok.filter(pl.Series(present))
    return s1tok.select(["country", "nt"]).explode("nt").drop_nulls("nt").group_by(["country", "nt"]).len() \
        .rename({"nt": "tok", "len": "df"}).with_columns(pl.col("df").cast(pl.Float32))


def list_feats(idx, sc, a, b, dropped, name):
    I = np.asarray(idx[b]).astype(np.int64)
    S = np.asarray(sc[b]).astype(np.float32)
    valid = I >= 0
    if dropped is not None:
        valid &= ~dropped[np.maximum(I, 0)]
    S = np.where(valid, S, -1.0)
    top = S.max(1)
    srt = -np.sort(-S, 1)
    second = np.where(srt.shape[1] > 1, srt[:, 1], -1.0) if srt.shape[1] > 1 else np.full(len(S), -1.0)
    hit = valid & (I == a[:, None])
    psc = np.where(hit, S, -1.0).max(1)
    psc = np.where(psc < 0, 0.0, psc)
    top0 = np.maximum(top, 0.0)
    return {f"v5_{name}_top": top0, f"v5_{name}_n05": ((S >= top[:, None] - 0.05) & valid).sum(1),
            f"v5_{name}_n15": ((S >= top[:, None] - 0.15) & valid).sum(1),
            f"v5_{name}_gap12": np.where(second >= 0, top0 - np.maximum(second, 0), top0),
            f"v5_{name}_sc": psc, f"v5_{name}_gap": top0 - psc}


def sets(x, y, pre):
    inter = pl.col(x).list.set_intersection(pl.col(y)).list.len().cast(pl.Float32)
    union = pl.col(x).list.set_union(pl.col(y)).list.len().cast(pl.Float32)
    nx, ny = pl.col(x).list.len().cast(pl.Float32), pl.col(y).list.len().cast(pl.Float32)
    both = (nx > 0) & (ny > 0)
    return [pl.when(both).then(inter / union).otherwise(-1.0).alias(f"{pre}_jac"),
            pl.when(both).then(inter / nx).otherwise(-1.0).alias(f"{pre}_c12"),
            pl.when(both).then(inter / ny).otherwise(-1.0).alias(f"{pre}_c21")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="feature parts under feats/: c2_train_sim19 or c2_test")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--cosvar", default="none", help="test only: replace cos_e3 features by p2/cosvar/c2_test_<var>")
    ap.add_argument("--absent_code", type=int, default=-1,
                    help="drop_<code>.npy = S1 absent from the universe (default: <N> of 'sim<N>' in --src, else none)")
    ap.add_argument("--out", default="", help="output table under feats/ (default: --src with c2_ -> c2v5_)")
    args = ap.parse_args()
    t0 = time.time()
    split = "test" if "test" in args.src else "train"
    out_name = args.out or args.src.replace("c2_", "c2v5_", 1)
    out_dir = os.path.join(CACHE, "feats", out_name)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.src, "part_*.parquet")))
    assert files, f"no feature parts under feats/{args.src}"
    dropped = None
    code = args.absent_code if args.absent_code >= 0 else (int(args.src.split("sim")[-1]) if "sim" in args.src else -1)
    if code >= 0 and split == "train":
        dropped = np.load(os.path.join(CACHE, f"drop_{code}.npy"))
    s1, rr = load_norm(split)
    T1, TR = token_lists(s1), token_lists(rr)
    # V5 counted every S1 of the file; S1 absent from a simulated universe are not in its index, so they are left out
    DF = df_table(T1, None if dropped is None else ~dropped)
    # per-row name specificity
    def row_df(T, pre):
        e = T.with_row_index("i").select(["i", "country", "nt"]).explode("nt").join(
            DF, left_on=["country", "nt"], right_on=["country", "tok"], how="left").with_columns(pl.col("df").fill_null(0.0))
        g = e.group_by("i").agg(pl.col("df").min().alias("mn"), pl.col("df").log1p().mean().alias("me"))
        base = pl.DataFrame({"i": np.arange(T.height, dtype=np.uint32)})
        g = base.join(g.with_columns(pl.col("i").cast(pl.UInt32)), on="i", how="left").sort("i")
        return np.log1p(g["mn"].fill_null(0).to_numpy()).astype(np.float32), g["me"].fill_null(0).to_numpy().astype(np.float32)
    s1_min, s1_mean = row_df(T1, "s1")
    rec_min, _ = row_df(TR, "rec")
    bdir = os.path.join(CACHE, "blocking", f"{args.btag}_{split}")
    L = {l: (np.load(os.path.join(bdir, f"rid_{l}_idx.npy"), mmap_mode="r"), np.load(os.path.join(bdir, f"rid_{l}_sc.npy"), mmap_mode="r"))
         for l in ("name", "addr")}
    ov = None
    if split == "test" and args.cosvar != "none":
        from phase2.p2lib import apply_override
        ov = pl.read_parquet(os.path.join(CACHE, "p2", "cosvar", f"c2_test_{args.cosvar}.parquet"))
    for i, f in enumerate(files):
        d = pl.read_parquet(f)
        if ov is not None:
            d = apply_override(d, ov)
        a, b = d["a"].to_numpy().astype(np.int64), d["b"].to_numpy().astype(np.int64)
        new = {}
        for l, (idx, sc) in L.items():
            new.update(list_feats(idx, sc, a, b, dropped, l))
        new["v5_s1_mindf"], new["v5_s1_meandf"], new["v5_rec_mindf"] = s1_min[a], s1_mean[a], rec_min[b]
        P = pl.DataFrame({"k": np.arange(len(a), dtype=np.uint32), "country": T1["country"].gather(a),
                          "n1": T1["nt"].gather(a), "n2": TR["nt"].gather(b), "s1": T1["st"].gather(a), "s2": TR["st"].gather(b)})
        sh = P.select(["k", "country", pl.col("n1").list.set_intersection(pl.col("n2")).alias("sh")]).explode("sh").drop_nulls("sh")
        sh = sh.join(DF, left_on=["country", "sh"], right_on=["country", "tok"], how="left").with_columns(pl.col("df").fill_null(0.0))
        g = sh.group_by("k").agg(pl.col("df").min().log1p().alias("mn"), (pl.col("df") <= RARE).sum().alias("nr"))
        g = P.select("k").join(g, on="k", how="left").sort("k")
        new["v5_shared_mindf"] = g["mn"].fill_null(-1.0).to_numpy().astype(np.float32)
        new["v5_n_shared_rare"] = g["nr"].fill_null(0).to_numpy().astype(np.float32)
        st = P.select(sets("s1", "s2", "v5_st"))
        j1 = P["s1"].list.join(" ").to_list()
        j2 = P["s2"].list.join(" ").to_list()
        tset = cpdist(j1, j2, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
        emp = (P["s1"].list.len() == 0).to_numpy() | (P["s2"].list.len() == 0).to_numpy()
        new["v5_st_tset"] = np.where(emp, -1.0, tset).astype(np.float32)
        d = d.with_columns([pl.Series(k, np.asarray(v).astype(np.float32)) for k, v in new.items()]).hstack(st)
        d.write_parquet(os.path.join(out_dir, f"part_{i:03d}.parquet"))
        print(f"  part {i}: {d.height} pairs, {time.time() - t0:.0f}s", flush=True)
    cols = [c for c in d.columns if c.startswith("v5_")]
    os.makedirs(os.path.join(RESULTS, "v5"), exist_ok=True)
    json.dump({"src": args.src, "out": out_dir, "new_columns": cols, "cosvar": args.cosvar, "absent_code": code,
               "runtime_s": time.time() - t0},
              open(os.path.join(RESULTS, "v5", f"extra_feats_{out_name}.json"), "w"), indent=1)
    print("new columns:", cols, flush=True)


if __name__ == "__main__":
    main()
