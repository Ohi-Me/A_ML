"""Micro benchmark for the retrieval kernel: which way of scoring + top-k is fastest on the MIG slice."""
import os
import sys
import time

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import CACHE, NORM  # noqa: E402

dev = gpu_init()
cdir = os.path.join(CACHE, "blocking", "b1_train")
name = sp.load_npz(os.path.join(cdir, "tfidf_name.npz")).tocsr()
s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_train_s1.parquet"), columns=["country"])
n1 = s1.height
i1 = np.where(s1["country"].to_numpy() == "India")[0]
X = name[i1]
Q = name[n1:n1 + 20000]
print("index", X.shape, X.nnz, "queries", Q.shape, flush=True)


def to_csr(m, dtype=torch.float32):
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data).to(dtype), size=m.shape, device=dev)


def dense_T(m_rows, dtype=torch.float32):
    coo = m_rows.tocoo()
    d = torch.zeros((m_rows.shape[1], m_rows.shape[0]), device=dev, dtype=dtype)
    d[torch.from_numpy(coo.col.astype(np.int64)).to(dev), torch.from_numpy(coo.row.astype(np.int64)).to(dev)] = \
        torch.from_numpy(coo.data).to(dev).to(dtype)
    return d


Xt = to_csr(X)
for B in (512, 1024, 2048):
    for mode in ("topk_dim0", "topk_T"):
        torch.cuda.synchronize()
        t = {"build": 0, "mm": 0, "topk": 0}
        n = 0
        for s in range(0, 8 * B, B):
            t0 = time.time()
            D = dense_T(Q[s:s + B])
            torch.cuda.synchronize()
            t1 = time.time()
            S = torch.sparse.mm(Xt, D)
            torch.cuda.synchronize()
            t2 = time.time()
            if mode == "topk_dim0":
                v, i = torch.topk(S, 20, dim=0)
            else:
                v, i = torch.topk(S.T.contiguous(), 20, dim=1)
            i.cpu()
            torch.cuda.synchronize()
            t3 = time.time()
            t["build"] += t1 - t0
            t["mm"] += t2 - t1
            t["topk"] += t3 - t2
            n += B
            del S, D
        print(B, mode, {k: round(v / n * 1e6, 2) for k, v in t.items()}, "us/query", flush=True)

# fp16 sparse
try:
    Xh = to_csr(X, torch.float16)
    D = dense_T(Q[:1024], torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        S = torch.sparse.mm(Xh, D)
    torch.cuda.synchronize()
    print("fp16 spmm", round((time.time() - t0) / 5 / 1024 * 1e6, 2), "us/query", flush=True)
except Exception as e:  # noqa: BLE001
    print("fp16 spmm failed", e)

# dense matmul reference: random projection to 256 dims
W = torch.randn(X.shape[1], 256, device=dev, dtype=torch.float16) / 16
E = torch.sparse.mm(to_csr(X), W.float()).half()
q = torch.randn(4096, 256, device=dev, dtype=torch.float16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(5):
    S = q @ E.T
    v, i = torch.topk(S, 20, dim=1)
torch.cuda.synchronize()
print("dense 256-d matmul+topk", round((time.time() - t0) / 5 / 4096 * 1e6, 2), "us/query", flush=True)
