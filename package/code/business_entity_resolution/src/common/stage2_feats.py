"""Stage-2 feature builders shared by code/xgboost/stage2.py and code/final/predict_test_v2.py."""
import numpy as np
import polars as pl
import torch


def context_features(D, conf_thr=0.5):
    """D: a, b, p (+ row id r). Returns frame r + context columns."""
    D = D.with_columns([
        pl.col("p").max().over("b").alias("s2_b_p1"),
        pl.col("p").rank("ordinal", descending=True).over("b").cast(pl.Int32).alias("s2_b_rank"),
        pl.col("p").sum().over("a").alias("s2_a_psum"),
        (pl.col("p") > conf_thr).sum().over("a").cast(pl.Int32).alias("s2_a_nconf"),
        pl.col("p").rank("ordinal", descending=True).over("a").cast(pl.Int32).alias("s2_a_rank"),
        pl.len().over("a").cast(pl.Int32).alias("s2_a_n"),
    ])
    sec_b = D.filter(pl.col("s2_b_rank") == 2).select(["b", pl.col("p").alias("s2_b_p2")])
    top_a = D.filter(pl.col("s2_a_rank") <= 2).select(["a", "s2_a_rank", "p"]).pivot(
        on="s2_a_rank", index="a", values="p").rename({"1": "_a1", "2": "_a2"})
    D = D.join(sec_b, on="b", how="left").join(top_a, on="a", how="left")
    D = D.with_columns([
        pl.col("s2_b_p2").fill_null(0.0),
        (pl.col("p") - pl.col("s2_b_p1")).alias("s2_b_gap"),
        pl.when(pl.col("s2_a_rank") == 1).then(pl.col("_a2")).otherwise(pl.col("_a1")).fill_null(0.0).alias("s2_a_other_max"),
    ]).drop(["_a1", "_a2"])
    D = D.with_columns((pl.col("p") - pl.col("s2_b_p2")).alias("s2_b_margin"),
                       (pl.col("s2_a_psum") - pl.col("p")).alias("s2_a_psum_other"))
    # how many records choose this S1 as their best with p > thr
    best = D.filter((pl.col("s2_b_rank") == 1) & (pl.col("p") > conf_thr)).group_by("a").len().rename({"len": "s2_a_claimed"})
    D = D.join(best, on="a", how="left").with_columns(pl.col("s2_a_claimed").fill_null(0).cast(pl.Int32))
    return D


def sibling_features(D, E, NAME, ADDR, n1, dev, conf_thr=0.5, chunk_a=60_000):
    """For each pair (a, b): similarity of record b to the other confident records b' of the same S1."""
    conf = D.filter(pl.col("p") > conf_thr).select(["a", pl.col("b").alias("b2")])
    a_max = int(D["a"].max()) + 1
    outs = []
    for s in range(0, a_max, chunk_a):
        d = D.filter((pl.col("a") >= s) & (pl.col("a") < s + chunk_a)).select(["r", "a", "b"])
        c = conf.filter((pl.col("a") >= s) & (pl.col("a") < s + chunk_a))
        j = d.join(c, on="a").filter(pl.col("b") != pl.col("b2"))
        if j.height == 0:
            continue
        b1 = j["b"].to_numpy().astype(np.int64) + n1
        b2 = j["b2"].to_numpy().astype(np.int64) + n1
        ce = np.zeros(j.height, np.float32)
        for t in range(0, j.height, 8_000_000):
            x = torch.from_numpy(b1[t:t + 8_000_000]).to(dev)
            y = torch.from_numpy(b2[t:t + 8_000_000]).to(dev)
            ce[t:t + 8_000_000] = (E[x].float() * E[y].float()).sum(1).cpu().numpy()
        cn = np.asarray(NAME[b1].multiply(NAME[b2]).sum(axis=1)).ravel().astype(np.float32)
        ca = np.asarray(ADDR[b1].multiply(ADDR[b2]).sum(axis=1)).ravel().astype(np.float32)
        g = j.select("r").with_columns(pl.Series("ce", ce), pl.Series("cn", cn), pl.Series("ca", ca)).group_by("r").agg(
            pl.col("ce").max().alias("sib_emb_max"), pl.col("ce").mean().alias("sib_emb_mean"),
            pl.col("cn").max().alias("sib_name_max"), pl.col("ca").max().alias("sib_addr_max"),
            pl.len().cast(pl.Int32).alias("sib_n"))
        outs.append(g)
        print(f"    siblings a<{s + chunk_a}: {j.height} record pairs", flush=True)
    S = pl.concat(outs)
    return D.join(S, on="r", how="left").with_columns(
        [pl.col(c).fill_null(-1.0) for c in ("sib_emb_max", "sib_emb_mean", "sib_name_max", "sib_addr_max")] +
        [pl.col("sib_n").fill_null(0)])
