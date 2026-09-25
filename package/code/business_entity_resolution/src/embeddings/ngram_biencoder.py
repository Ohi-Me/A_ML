"""Embedding experiment E1: learned n-gram bi-encoder for blocking (and as a pair feature).

Input per record = the hashed TF-IDF vectors made by blocking (name: char 3-grams + words, address: tokens).
Encoder: EmbeddingBag(sum, weighted by tf-idf) for name and address -> MLP -> 256-d unit vector. Same encoder for
S1 and S2/S3 records. Trained with symmetric InfoNCE on true pairs of folds 1-4 only; in-batch negatives come
from the same country/state bucket so they are hard. Retrieval = fp16 matmul + top-k per country.

  python code/embeddings/ngram_biencoder.py --mode train --tag e1          (train split: fit + embed + recall)
  python code/embeddings/ngram_biencoder.py --mode embed --tag e1 --split test --model_from e1   (test split)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402


class GCSR:
    """A CSR matrix kept on the GPU, with fast row-batch extraction for EmbeddingBag."""

    def __init__(self, m, dev):
        self.indptr = torch.from_numpy(m.indptr.astype(np.int64)).to(dev)
        self.indices = torch.from_numpy(m.indices.astype(np.int64)).to(dev)
        self.data = torch.from_numpy(m.data.astype(np.float32)).to(dev)
        self.dev = dev

    def batch(self, rows, drop=0.0):
        st = self.indptr[rows]
        ln = self.indptr[rows + 1] - st
        offsets = torch.cumsum(ln, 0) - ln
        tot = int(ln.sum())
        idx = torch.repeat_interleave(st - offsets, ln, output_size=tot) + torch.arange(tot, device=self.dev)
        w = self.data[idx]
        if drop > 0:   # n-gram dropout: a cheap augmentation against memorizing exact n-gram sets
            w = w * (torch.rand_like(w) >= drop).float() / (1 - drop)
        return self.indices[idx], offsets, w


class Encoder(nn.Module):
    def __init__(self, hn, ha, d=256, h=768, out=256):
        super().__init__()
        self.en = nn.EmbeddingBag(hn, d, mode="sum", sparse=True)
        self.ea = nn.EmbeddingBag(ha, d, mode="sum", sparse=True)
        nn.init.normal_(self.en.weight, std=0.05)
        nn.init.normal_(self.ea.weight, std=0.05)
        self.mlp = nn.Sequential(nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, h), nn.GELU(), nn.Linear(h, out))

    def forward(self, nb, ab, has_addr):
        vn = self.en(*nb[:2], per_sample_weights=nb[2])
        va = self.ea(*ab[:2], per_sample_weights=ab[2])
        z = self.mlp(torch.cat([vn, va, has_addr[:, None]], 1))
        return F.normalize(z, dim=-1)


def encode_rows(model, NM, AD, HAS, rows, bs=65536):
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            r = rows[s:s + bs]
            out.append(model(NM.batch(r), AD.batch(r), HAS[r]).half())
    model.train()
    return torch.cat(out)


def topk_dense(Q, E, K, B=None):
    """Q (nq x d) fp16, E (N x d) fp16 on GPU -> idx (nq x K) int32, score fp16 on CPU."""
    K = min(K, E.shape[0])
    B = B or max(256, int(4e9 // (E.shape[0] * 2)))
    idx = np.zeros((Q.shape[0], K), np.int32)
    sc = np.zeros((Q.shape[0], K), np.float16)
    for s in range(0, Q.shape[0], B):
        S = Q[s:s + B] @ E.T
        v, i = torch.topk(S, K, dim=1)
        idx[s:s + B] = i.int().cpu().numpy()
        sc[s:s + B] = v.cpu().numpy()
    return idx, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "embed"])
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--btag", default="b1", help="blocking tag whose tf-idf matrices are the inputs")
    ap.add_argument("--model_from", default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--final", action="store_true", help="train on all folds (for the test run)")
    ap.add_argument("--drop", type=float, default=0.0, help="n-gram dropout during training")
    ap.add_argument("--fit_country", default="", help="train only on pairs of this country (country-shift test)")
    ap.add_argument("--train_only", action="store_true", help="fit and save the model, skip embedding the split")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    bdir = os.path.join(CACHE, "blocking", f"{args.btag}_{args.split}")
    edir = os.path.join(CACHE, "emb", f"{args.tag}_{args.split}")
    rdir = os.path.join(RESULTS, "embeddings", args.tag)
    mdir = os.path.join(CACHE, "emb", "models")
    for d in (edir, rdir, mdir):
        os.makedirs(d, exist_ok=True)

    with Timer("load inputs to GPU"):
        name = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        addr = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        hn, ha = name.shape[1], addr.shape[1]
        HAS = torch.from_numpy((np.diff(addr.indptr) > 0).astype(np.float32)).to(dev)
        NM, AD = GCSR(name, dev), GCSR(addr, dev)
        n_all = name.shape[0]
        del name, addr
        s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s1.parquet"), columns=["id", "country", "state"])
        rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s{k}.parquet"),
                                        columns=["id", "country", "state"]) for k in (2, 3)])
        n1 = s1.height
    print("rows", n_all, "S1", n1, gpu_mem(), flush=True)
    model = Encoder(hn, ha).to(dev)
    mpath = os.path.join(mdir, f"{args.model_from or args.tag}.pt")

    if args.mode == "train":
        pairs, _ = load_gt()
        s1_map = pl.DataFrame({"s1": s1["id"], "a": np.arange(n1, dtype=np.int64)})
        r_map = pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height, dtype=np.int64)})
        P = pairs.join(s1_map, on="s1").join(r_map, on="rid").with_columns(
            pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
        if not args.final:
            P = P.filter(pl.col("fold") != VAL_FOLD)
        P = P.join(s1.select(pl.col("country"), pl.col("state"), pl.int_range(pl.len()).alias("a").cast(pl.Int64)), on="a")
        if args.fit_country:
            P = P.filter(pl.col("country") == args.fit_country)
        print("training pairs", P.height, flush=True)
        dense_params = [p for n_, p in model.named_parameters() if not n_.startswith(("en.", "ea."))]
        opt_d = torch.optim.AdamW(dense_params, lr=args.lr, weight_decay=0.01)
        opt_s = torch.optim.SparseAdam(list(model.en.parameters()) + list(model.ea.parameters()), lr=args.lr)
        log_t = torch.tensor(np.log(1 / 0.05), device=dev, requires_grad=True)
        opt_t = torch.optim.Adam([log_t], lr=1e-3)
        a_np, b_np = P["a"].to_numpy(), P["b"].to_numpy()
        bucket = (P["country"] + "|" + P["state"]).to_numpy()
        _, bucket_id = np.unique(bucket, return_inverse=True)
        hist = []
        for ep in range(args.epochs):
            # one record per S1 per epoch (avoids duplicate-positive clashes in a batch), batches from one bucket
            perm = np.random.permutation(len(a_np))
            _, first = np.unique(a_np[perm], return_index=True)
            sel = perm[first]
            sel = sel[np.lexsort((np.random.rand(len(sel)), bucket_id[sel]))]
            batches = [sel[i:i + args.bs] for i in range(0, len(sel), args.bs)]
            np.random.shuffle(batches)
            tl, n = 0.0, 0
            for bi in batches:
                ra = torch.from_numpy(a_np[bi]).to(dev)
                rb = torch.from_numpy(b_np[bi] + n1).to(dev)
                za = model(NM.batch(ra, args.drop), AD.batch(ra, args.drop), HAS[ra])
                zb = model(NM.batch(rb, args.drop), AD.batch(rb, args.drop), HAS[rb])
                logits = za @ zb.T * log_t.exp().clamp(max=100)
                lab = torch.arange(len(bi), device=dev)
                loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
                opt_d.zero_grad()
                opt_s.zero_grad()
                opt_t.zero_grad()
                loss.backward()
                opt_d.step()
                opt_s.step()
                opt_t.step()
                tl += loss.item() * len(bi)
                n += len(bi)
            hist.append({"epoch": ep, "loss": tl / n, "temp": float(1 / log_t.exp()), "time_s": time.time() - t0})
            print(hist[-1], flush=True)
        torch.save(model.state_dict(), mpath)
        del opt_d, opt_s, opt_t, P, pairs, s1_map, r_map, a_np, b_np, bucket, bucket_id
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        model.load_state_dict(torch.load(mpath, map_location=dev))
        hist = []

    if args.train_only:
        json.dump({"history": hist, "args": vars(args), "runtime_s": time.time() - t0},
                  open(os.path.join(rdir, f"report_trainonly_{args.split}.json"), "w"), indent=1)
        return
    with Timer("embed all rows"):
        E = encode_rows(model, NM, AD, HAS, torch.arange(n_all, device=dev))
        del NM, AD
        torch.cuda.empty_cache()
        mm = np.lib.format.open_memmap(os.path.join(edir, "emb.npy"), mode="w+", dtype=np.float16, shape=tuple(E.shape))
        for s in range(0, E.shape[0], 1_000_000):
            mm[s:s + 1_000_000] = E[s:s + 1_000_000].cpu().numpy()
        mm.flush()
        del mm
    with Timer("dense retrieval per country"):
        s1c, rc = s1["country"].to_numpy(), rr["country"].to_numpy()
        ridx = np.full((rr.height, args.k), -1, np.int32)
        rsc = np.zeros((rr.height, args.k), np.float16)
        for c in sorted(set(s1c.tolist())):
            i1 = np.where(s1c == c)[0]
            ir = np.where(rc == c)[0]
            if len(ir) == 0:
                continue
            E1 = E[torch.from_numpy(i1).to(dev)]
            Q = E[torch.from_numpy(ir + n1).to(dev)]
            idx, sc = topk_dense(Q, E1, args.k)
            ridx[ir, :idx.shape[1]] = i1[idx]
            rsc[ir, :idx.shape[1]] = sc
            del E1, Q
        np.save(os.path.join(edir, "rid_emb_idx.npy"), ridx)
        np.save(os.path.join(edir, "rid_emb_sc.npy"), rsc)
    rep = {"history": hist, "args": vars(args)}
    if args.split == "train":
        with Timer("recall"):
            pairs, _ = load_gt()
            s1_map = pl.DataFrame({"s1": s1["id"], "a": np.arange(n1, dtype=np.int64)})
            r_map = pl.DataFrame({"rid": rr["id"], "b": np.arange(rr.height, dtype=np.int64)})
            g = pairs.join(s1_map, on="s1").join(r_map, on="rid").with_columns(
                pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
            for nm, gg in (("val", g.filter(pl.col("fold") == VAL_FOLD)), ("train_folds", g.filter(pl.col("fold") != VAL_FOLD))):
                a, b = gg["a"].to_numpy(), gg["b"].to_numpy()
                hit = ridx[b] == a[:, None]
                rk = np.where(hit.any(1), hit.argmax(1), 999)
                rep[nm] = {f"recall@{k}": float((rk < k).mean()) for k in (1, 2, 3, 5, 10, 20)}
                # also against tf-idf comb list of the blocking run, for the union
                bi = os.path.join(CACHE, "blocking", f"{args.btag}_{args.split}", "rid_comb_idx.npy")
                if os.path.exists(bi):
                    bl = np.load(bi, mmap_mode="r")
                    h2 = np.asarray(bl[b]) == a[:, None]
                    rk2 = np.where(h2.any(1), h2.argmax(1), 999)
                    for k in (3, 5, 10, 20):
                        rep[nm][f"union_tfidf_comb@{k}"] = float(((rk < k) | (rk2 < k)).mean())
                print(nm, rep[nm], flush=True)
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    json.dump(rep, open(os.path.join(rdir, f"report_{args.mode}_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
