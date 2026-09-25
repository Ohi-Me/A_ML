"""Blocking experiment B1: hashed TF-IDF retrieval on the GPU (cuSPARSE SpMM), per country.

For every S2/S3 record we keep the top-K Source-1 entities of three lists (name cosine, address cosine,
name+address), and for every S1 entity the top-K records. Big arrays go to data/cache/blocking/<tag>/,
recall tables go to results/blocking/<tag>/.

  python code/blocking/sparse_tfidf.py --split train --tag b1
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

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402

H = 1 << 19          # hashed features per field
LISTS = ("name", "addr", "comb")


# ---------------------------------------------------------------------------------------------- featurization
def _hash_chunk(args):
    from sklearn.feature_extraction.text import HashingVectorizer
    kind, texts = args
    if kind == "name_char":
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=H, alternate_sign=False, norm=None,
                               lowercase=False, dtype=np.float32)
    else:   # words (name words, address tokens)
        hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", n_features=H, alternate_sign=False, norm=None,
                               lowercase=False, dtype=np.float32)
    return hv.transform(texts)


def hashed(texts, kind, step=500000):
    # one process (worker pools hang on this node: /dev/shm semaphores disappear)
    parts = [_hash_chunk((kind, texts[i:i + step])) for i in range(0, len(texts), step)]
    return sp.vstack(parts).tocsr()


def tfidf(m, df, n_docs):
    m = m.copy()
    m.data = np.log1p(m.data)
    idf = (np.log((1 + n_docs) / (1 + df)) + 1).astype(np.float32)
    m = m.multiply(idf[None, :]).tocsr()
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return sp.diags(1 / norms).dot(m).tocsr().astype(np.float32)


def to_torch_csr(m, dev):
    return torch.sparse_csr_tensor(torch.from_numpy(m.indptr.astype(np.int64)), torch.from_numpy(m.indices.astype(np.int64)),
                                   torch.from_numpy(m.data), size=m.shape, device=dev)


def dense_T(m_rows, dev):
    """rows of a scipy CSR -> dense (H x B) on GPU."""
    coo = m_rows.tocoo()
    d = torch.zeros((m_rows.shape[1], m_rows.shape[0]), device=dev, dtype=torch.float32)
    d[torch.from_numpy(coo.col.astype(np.int64)).to(dev), torch.from_numpy(coo.row.astype(np.int64)).to(dev)] = \
        torch.from_numpy(coo.data).to(dev)
    return d


def retrieve(Xn, Xa, Qn, Qa, K, B, dev, wn=1.0, wa=1.0):
    """Index (Xn, Xa: torch CSR, N x H), queries (scipy CSR). Returns per list: idx (nq x K int32), score fp16,
    plus the name / address cosine of the comb list.
    top-k runs on a contiguous fp16 (B x N) copy: top-k along the strided dim was 5x slower (bench_spmm.py)."""
    nq = Qn.shape[0]
    N = Xn.shape[0]
    K = min(K, N)
    out = {l: (np.zeros((nq, K), np.int32), np.zeros((nq, K), np.float16)) for l in LISTS}
    comb_parts = (np.zeros((nq, K), np.float16), np.zeros((nq, K), np.float16))
    t_start = time.time()
    for s in range(0, nq, B):
        e = min(s + B, nq)
        if (s // B) % 200 == 0:
            el = time.time() - t_start
            print(f"    {s}/{nq} queries, {el:.0f}s, eta {el / max(s, 1) * (nq - s):.0f}s", flush=True)
        D = dense_T(Qn[s:e], dev)
        SnT = torch.sparse.mm(Xn, D).T.contiguous().half()
        del D
        D = dense_T(Qa[s:e], dev)
        SaT = torch.sparse.mm(Xa, D).T.contiguous().half()
        del D
        for l, S in (("name", SnT), ("addr", SaT)):
            v, i = torch.topk(S, K, dim=1)
            out[l][0][s:e] = i.int().cpu().numpy()
            out[l][1][s:e] = v.cpu().numpy()
        Sc = SnT * wn + SaT * wa
        v, i = torch.topk(Sc, K, dim=1)
        del Sc
        out["comb"][0][s:e] = i.int().cpu().numpy()
        out["comb"][1][s:e] = v.cpu().numpy()
        comb_parts[0][s:e] = SnT.gather(1, i).cpu().numpy()
        comb_parts[1][s:e] = SaT.gather(1, i).cpu().numpy()
        del SnT, SaT
    return out, comb_parts


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--norm", default=NORM)
    ap.add_argument("--k_rid", type=int, default=20)
    ap.add_argument("--k_s1", type=int, default=50)
    ap.add_argument("--no_s1side", action="store_true", help="skip the S1 -> record pass")
    args = ap.parse_args()
    dev = gpu_init()
    cdir = os.path.join(CACHE, "blocking", f"{args.tag}_{args.split}")
    rdir = os.path.join(RESULTS, "blocking", args.tag)
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    t0 = time.time()

    with Timer("load normalized"):
        s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s1.parquet"),
                             columns=["id", "country", "core", "am"])
        rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{args.norm}_{args.split}_s{k}.parquet"),
                                        columns=["id", "country", "core", "am"]) for k in (2, 3)])
    with Timer("hash features"):
        allc = pl.concat([s1, rr])
        n_docs = allc.height
        mats = {}
        for f, kind, col in (("nc", "name_char", "core"), ("nw", "word", "core"), ("aw", "word", "am")):
            m = hashed(allc[col].to_list(), kind)
            df = np.bincount(m.indices, minlength=H).astype(np.float32)
            mats[f] = tfidf(m, df, n_docs)
            print(f, "nnz/doc", round(mats[f].nnz / n_docs, 1), flush=True)
        # name vector = char 3-grams + words (words at half weight), renormalized
        name = sp.hstack([mats["nc"], 0.5 * mats["nw"]]).tocsr()
        nn = np.sqrt(np.asarray(name.multiply(name).sum(axis=1)).ravel())
        nn[nn == 0] = 1
        name = sp.diags(1 / nn).dot(name).tocsr().astype(np.float32)
        addr = mats["aw"]
        del mats, allc
    n1 = s1.height
    with Timer("save tfidf matrices"):
        sp.save_npz(os.path.join(cdir, "tfidf_name.npz"), name, compressed=False)
        sp.save_npz(os.path.join(cdir, "tfidf_addr.npz"), addr, compressed=False)
    s1_country = s1["country"].to_numpy()
    rr_country = rr["country"].to_numpy()
    s1.select("id").write_parquet(os.path.join(cdir, "s1_ids.parquet"))
    rr.select("id").write_parquet(os.path.join(cdir, "rid_ids.parquet"))
    nr = rr.height
    del s1, rr
    countries = sorted(set(s1_country.tolist()))
    print("countries in S1:", countries, flush=True)

    # record -> S1 lists, per country, written straight into disk-backed arrays (global row numbers)
    K1 = args.k_rid

    def mm(name_, dtype, fill):
        m = np.lib.format.open_memmap(os.path.join(cdir, name_), mode="w+", dtype=dtype, shape=(nr, K1))
        m[:] = fill
        return m
    rid_res = {l: (mm(f"rid_{l}_idx.npy", np.int32, -1), mm(f"rid_{l}_sc.npy", np.float16, 0)) for l in LISTS}
    rid_parts = (mm("rid_comb_cosname.npy", np.float16, 0), mm("rid_comb_cosaddr.npy", np.float16, 0))
    timing = {}
    for c in countries:
        i1 = np.where(s1_country == c)[0]
        ir = np.where(rr_country == c)[0]
        if len(ir) == 0:
            continue
        with Timer(f"{c}: {len(ir)} records -> {len(i1)} S1") as tm:
            Xn, Xa = to_torch_csr(name[i1], dev), to_torch_csr(addr[i1], dev)
            out, parts = retrieve(Xn, Xa, name[ir + n1], addr[ir + n1], K1, 2048, dev)
            k = out["name"][0].shape[1]
            for l in LISTS:
                rid_res[l][0][ir, :k] = i1[out[l][0]]
                rid_res[l][1][ir, :k] = out[l][1]
            rid_parts[0][ir, :k], rid_parts[1][ir, :k] = parts
            for m_ in list(rid_res.values()):
                m_[0].flush(), m_[1].flush()
            del Xn, Xa, out, parts
            torch.cuda.empty_cache()
        timing[f"{c}_rid2s1_s"] = tm.s
    print("gpu", gpu_mem(), flush=True)
    del name, addr
    rid_res = {l: (np.load(os.path.join(cdir, f"rid_{l}_idx.npy"), mmap_mode="r"),) for l in LISTS}
    s1_res = None
    json.dump(timing, open(os.path.join(rdir, f"timing_{args.split}.json"), "w"), indent=1)

    if args.split != "train":
        return
    # ------------------------------------------------------------------------------------------ recall
    with Timer("recall analysis"):
        pairs, per = load_gt()
        s1_ids = pl.read_parquet(os.path.join(cdir, "s1_ids.parquet"))["id"]
        rid_ids = pl.read_parquet(os.path.join(cdir, "rid_ids.parquet"))["id"]
        m1 = pl.DataFrame({"s1": s1_ids, "a": np.arange(len(s1_ids), dtype=np.int32)})
        mr = pl.DataFrame({"rid": rid_ids, "b": np.arange(len(rid_ids), dtype=np.int32)})
        p = pairs.join(m1, on="s1").join(mr, on="rid").with_columns(
            pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64).alias("fold"))
        rep = {"timing_s": timing}
        for fold_name, pf in (("val", p.filter(pl.col("fold") == VAL_FOLD)), ("all", p)):
            a = pf["a"].to_numpy()
            b = pf["b"].to_numpy()
            r = {}
            ranks = {}
            for l in LISTS:
                hit = np.asarray(rid_res[l][0][b]) == a[:, None]
                rk = np.where(hit.any(1), hit.argmax(1), 999)
                ranks[l] = rk
                for k in (1, 2, 3, 5, 10, 20):
                    r[f"rid2s1_{l}@{k}"] = float((rk < k).mean())
            for k in (1, 2, 3, 5, 10, 20):
                r[f"rid2s1_union@{k}"] = float(np.mean((ranks["name"] < k) | (ranks["addr"] < k) | (ranks["comb"] < k)))
            r["n_pairs"] = int(len(a))
            rep[fold_name] = r
            print(fold_name, json.dumps(r, indent=0), flush=True)
        rep["total_s"] = time.time() - t0
        json.dump(rep, open(os.path.join(rdir, "recall.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
