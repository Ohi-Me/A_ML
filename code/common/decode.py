"""Expected-F0.5 decoding per Source-1 entity.

1. every record goes to its best S1 (the ground truth never links a record to two S1);
2. for each S1, sort its assigned records by calibrated probability p1 >= p2 >= ...;
3. pick k (0..K) that maximizes E[F0.5] when predicting the top-k records.
   With TP true among the k predicted and T true in total: F0.5 = 1.25 TP / (0.25 T + k) for k > 0,
   and for k = 0 the entity scores 1 only if T = 0. The expectation is taken by Monte Carlo on the GPU,
   treating the records' labels as independent Bernoulli(p). A small 'extra' mass models true matches
   outside the candidate set (they add to T but can never be predicted)."""
import numpy as np
import polars as pl
import torch


def assign(D, floor=0.01):
    """D: a, b, p. Returns records kept at their best S1: a, b, p (p >= floor)."""
    best = D.sort(["b", "p"], descending=[False, True]).group_by("b", maintain_order=True).agg(
        pl.col("a").first(), pl.col("p").first())
    return best.filter(pl.col("p") >= floor)


def expected_f_decode(A, K=12, M=2048, extra=0.0, gamma=1.0, seed=0, dev=None, chunk=200_000):
    """A: a, b, p (one row per kept record). Returns chosen (a, b) and per S1 the chosen k and its expected F0.5
    (S1 without any kept record are not listed: their expected F0.5 is 1 under the model)."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")      # cpu only in smoke tests
    A = A.with_columns((pl.col("p") ** gamma).alias("q")).sort(["a", "q"], descending=[False, True])
    A = A.with_columns(pl.int_range(pl.len()).over("a").alias("pos")).filter(pl.col("pos") < K)
    g = A.group_by("a", maintain_order=True).agg(pl.col("q"), pl.col("b"))
    a_ids = g["a"].to_numpy()
    n = len(a_ids)
    P = np.zeros((n, K), np.float32)
    lens = g["q"].list.len().to_numpy().astype(np.int64)
    flat = np.concatenate(g["q"].to_list()).astype(np.float32) if n else np.zeros(0, np.float32)
    rows = np.repeat(np.arange(n), lens)
    cols = np.concatenate([np.arange(l) for l in lens]) if n else np.zeros(0, int)
    P[rows, cols] = flat
    best_k = np.zeros(n, np.int64)
    best_ef = np.zeros(n, np.float32)
    gen = torch.Generator(device=dev).manual_seed(seed)
    kk = torch.arange(1, K + 1, device=dev, dtype=torch.float32)
    for s in range(0, n, chunk):
        p = torch.from_numpy(P[s:s + chunk]).to(dev)                         # (c, K)
        c = p.shape[0]
        ef = torch.zeros((c, K + 1), device=dev)
        for m0 in range(0, M, 256):
            mm = min(256, M - m0)
            u = torch.rand((c, mm, K), device=dev, generator=gen)
            y = (u < p[:, None, :]).float()                                   # (c, mm, K)
            T = y.sum(-1)                                                     # (c, mm)
            if extra > 0:
                T = T + (torch.rand((c, mm), device=dev, generator=gen) < extra).float()
            tp = torch.cumsum(y, -1)                                          # (c, mm, K) TP of top-k
            f = 1.25 * tp / (0.25 * T[..., None] + kk)                        # k = 1..K
            ef[:, 1:] += f.sum(1)
            ef[:, 0] += (T == 0).float().sum(1)
        ef /= M
        # never predict beyond the records the S1 actually has
        valid = torch.arange(K + 1, device=dev)[None, :] <= torch.from_numpy(lens[s:s + chunk]).to(dev)[:, None]
        ef[~valid] = -1
        mx, am = ef.max(1)
        best_k[s:s + chunk] = am.cpu().numpy()
        best_ef[s:s + chunk] = mx.cpu().numpy()
    keep = g.with_columns(pl.Series("k", best_k)).with_columns(pl.col("b").list.head(pl.col("k"))).explode("b") \
        .filter(pl.col("b").is_not_null()).select(["a", "b"])
    return keep, pl.DataFrame({"a": a_ids, "k": best_k, "ef": best_ef})
