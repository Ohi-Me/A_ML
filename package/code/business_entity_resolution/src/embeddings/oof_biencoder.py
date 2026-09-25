"""Embedding experiment E3: cross-fitted (out-of-fold) n-gram bi-encoders.

Why: a single encoder trained on folds 1-4 gives in-sample cosines on the pairs the pair model is fitted on and
out-of-sample cosines on validation/test pairs. Here model_f is trained without fold f, and
  - a training record queries the S1 index with the model of its own fold (unmatched records get a hashed fold),
  - the cosine feature of a candidate pair comes from the model of the S1's fold,
so every training feature is out-of-sample, like at test time.
Test split: retrieval with the all-folds model, cosine feature = mean over the 5 fold models.

  python code/embeddings/oof_biencoder.py --split train --tag e3
  python code/embeddings/oof_biencoder.py --split test --tag e3 --cands c1      (after the models exist)
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS, load_gt, NORM  # noqa: E402
from common.split import N_FOLDS, VAL_FOLD, fold_of  # noqa: E402
from embeddings.ngram_biencoder import GCSR, Encoder, encode_rows, topk_dense  # noqa: E402


def train_model(NM, AD, HAS, a_np, b_np, bucket_id, n1, args, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Encoder(NM_H[0], NM_H[1]).to(dev)
    dense_params = [p for n_, p in model.named_parameters() if not n_.startswith(("en.", "ea."))]
    opt_d = torch.optim.AdamW(dense_params, lr=args.lr, weight_decay=0.01)
    opt_s = torch.optim.SparseAdam(list(model.en.parameters()) + list(model.ea.parameters()), lr=args.lr)
    log_t = torch.tensor(np.log(1 / 0.05), device=dev, requires_grad=True)
    opt_t = torch.optim.Adam([log_t], lr=1e-3)
    for ep in range(args.epochs):
        perm = rng.permutation(len(a_np))
        _, first = np.unique(a_np[perm], return_index=True)
        sel = perm[first]
        sel = sel[np.lexsort((rng.random(len(sel)), bucket_id[sel]))]
        batches = [sel[i:i + args.bs] for i in range(0, len(sel), args.bs)]
        rng.shuffle(batches)
        tl = 0.0
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
        print(f"    epoch {ep} loss {tl / len(sel):.4f}", flush=True)
    del opt_d, opt_s, opt_t
    return model


NM_H = (0, 0)


def main():
    global NM_H
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--tag", default="e3")
    ap.add_argument("--btag", default="b1")
    ap.add_argument("--cands", default="", help="candidate table whose pair cosines are written")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--reuse", action="store_true", help="load saved fold models instead of training (second pass)")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    bdir = os.path.join(CACHE, "blocking", f"{args.btag}_{args.split}")
    edir = os.path.join(CACHE, "emb", f"{args.tag}_{args.split}")
    mdir = os.path.join(CACHE, "emb", "models")
    rdir = os.path.join(RESULTS, "embeddings", args.tag)
    for d in (edir, mdir, rdir):
        os.makedirs(d, exist_ok=True)
    with Timer("load inputs"):
        name = sp.load_npz(os.path.join(bdir, "tfidf_name.npz")).tocsr()
        addr = sp.load_npz(os.path.join(bdir, "tfidf_addr.npz")).tocsr()
        NM_H = (name.shape[1], addr.shape[1])
        HAS = torch.from_numpy((np.diff(addr.indptr) > 0).astype(np.float32)).to(dev)
        NM, AD = GCSR(name, dev), GCSR(addr, dev)
        n_all = name.shape[0]
        del name, addr
        s1 = pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s1.parquet"), columns=["id", "country", "state"])
        rr = pl.concat([pl.read_parquet(os.path.join(CACHE, f"norm_{NORM}_{args.split}_s{k}.parquet"),
                                        columns=["id", "country"]) for k in (2, 3)])
        n1, nr = s1.height, rr.height
        s1c, rc = s1["country"].to_numpy(), rr["country"].to_numpy()
    rep = {"args": vars(args)}
    if args.reuse or args.split != "train":
        ridx = np.load(os.path.join(edir, "rid_emb_idx.npy"), mmap_mode="r") if os.path.exists(
            os.path.join(edir, "rid_emb_idx.npy")) else None
        rsc = None
    else:
        ridx = np.lib.format.open_memmap(os.path.join(edir, "rid_emb_idx.npy"), mode="w+", dtype=np.int32, shape=(nr, args.k))
        rsc = np.lib.format.open_memmap(os.path.join(edir, "rid_emb_sc.npy"), mode="w+", dtype=np.float16, shape=(nr, args.k))
        ridx[:] = -1
    C = None
    if args.cands:
        C = pl.read_parquet(os.path.join(CACHE, "cands", f"{args.cands}_{args.split}.parquet"), columns=["a", "b"])
        pair_cos = np.zeros(C.height, np.float32)

    def retrieve(E, rec_rows):
        for c in sorted(set(s1c.tolist())):
            i1 = np.where(s1c == c)[0]
            ir = rec_rows[rc[rec_rows] == c]
            if len(ir) == 0:
                continue
            idx, sc = topk_dense(E[torch.from_numpy(ir + n1).to(dev)], E[torch.from_numpy(i1).to(dev)], args.k)
            ridx[ir, :idx.shape[1]] = i1[idx]
            rsc[ir, :idx.shape[1]] = sc

    def pair_cosine(E, sel):
        a = torch.from_numpy(C["a"].to_numpy()[sel].astype(np.int64)).to(dev)
        b = torch.from_numpy(C["b"].to_numpy()[sel].astype(np.int64) + n1).to(dev)
        out = np.zeros(len(a), np.float32)
        for s in range(0, len(a), 4_000_000):
            out[s:s + 4_000_000] = (E[a[s:s + 4_000_000]].float() * E[b[s:s + 4_000_000]].float()).sum(1).cpu().numpy()
        return out

    if args.split == "train":
        pairs, _ = load_gt()
        m1 = pl.DataFrame({"s1": s1["id"], "a": np.arange(n1, dtype=np.int64)})
        mr = pl.DataFrame({"rid": rr["id"], "b": np.arange(nr, dtype=np.int64)})
        P = pairs.join(m1, on="s1").join(mr, on="rid")
        s1_fold = np.array([fold_of(x) for x in s1["id"].to_list()], np.int8)
        a_all, b_all = P["a"].to_numpy(), P["b"].to_numpy()
        pf = s1_fold[a_all]
        # fold of every record: its true S1's fold, or a hash fold for unmatched records
        rec_fold = np.array([zlib.crc32(x.encode()) % N_FOLDS for x in rr["id"].to_list()], np.int8)
        rec_fold[b_all] = pf
        buck = (s1["country"] + "|" + s1["state"]).to_numpy()
        _, buck_id_s1 = np.unique(buck, return_inverse=True)
        a_fold = s1_fold[C["a"].to_numpy()] if C is not None else None
        for f in range(N_FOLDS):
            with Timer(f"fold model {f}"):
                tr = pf != f
                mp_ = os.path.join(mdir, f"{args.tag}_fold{f}.pt")
                if args.reuse and os.path.exists(mp_):
                    model = Encoder(*NM_H).to(dev)
                    model.load_state_dict(torch.load(mp_, map_location=dev))
                else:
                    model = train_model(NM, AD, HAS, a_all[tr], b_all[tr], buck_id_s1[a_all[tr]], n1, args, dev, seed=f)
                    torch.save(model.state_dict(), mp_)
                E = encode_rows(model, NM, AD, HAS, torch.arange(n_all, device=dev))
                del model
                torch.cuda.empty_cache()
                if not args.reuse:
                    retrieve(E, np.where(rec_fold == f)[0])
                if C is not None:
                    sel = np.where(a_fold == f)[0]
                    pair_cos[sel] = pair_cosine(E, sel)
                del E
                torch.cuda.empty_cache()
        if not args.reuse:
            ridx.flush()
            rsc.flush()
        # recall on val (records of fold 0 queried with model_0: same as a single model trained on folds 1-4)
        pv = P.filter(pl.Series(pf == VAL_FOLD))
        a, b = pv["a"].to_numpy(), pv["b"].to_numpy()
        hit = np.asarray(ridx[b]) == a[:, None]
        rk = np.where(hit.any(1), hit.argmax(1), 999)
        rep["val"] = {f"recall@{k}": float((rk < k).mean()) for k in (1, 3, 5, 10, 20)}
        pt = P.filter(pl.Series(pf != VAL_FOLD))
        a, b = pt["a"].to_numpy(), pt["b"].to_numpy()
        hit = np.asarray(ridx[b]) == a[:, None]
        rk = np.where(hit.any(1), hit.argmax(1), 999)
        rep["train_folds_oof"] = {f"recall@{k}": float((rk < k).mean()) for k in (1, 3, 5, 10, 20)}
        print(json.dumps(rep), flush=True)
    else:
        # test: cosine feature = mean over fold models; retrieval with the mean-similarity of all fold models
        if C is not None:
            for f in range(N_FOLDS):
                with Timer(f"test cosines with fold model {f}"):
                    model = Encoder(*NM_H).to(dev)
                    model.load_state_dict(torch.load(os.path.join(mdir, f"{args.tag}_fold{f}.pt"), map_location=dev))
                    E = encode_rows(model, NM, AD, HAS, torch.arange(n_all, device=dev))
                    del model
                    pair_cos += pair_cosine(E, np.arange(C.height)) / N_FOLDS
                    del E
                    torch.cuda.empty_cache()
    if C is not None:
        C.with_columns(pl.Series(f"cos_{args.tag}", pair_cos)).write_parquet(
            os.path.join(edir, f"paircos_{args.cands}.parquet"))
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    json.dump(rep, open(os.path.join(rdir, f"report_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
