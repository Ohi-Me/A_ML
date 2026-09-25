"""Local GNN on the candidate graph (learned stage-2 context).

Graph : bipartite, S1 nodes and S2/S3 record nodes; one edge per candidate pair, edge input = the blended stage-1
        logit + ~20 key pair features (standardized).
Model : edge encoder MLP -> 2 message-passing rounds. Each round pools the edges of every S1 and of every record
        (mean + max) and updates each edge from [edge, pooled S1 side, pooled record side], so an edge sees the
        record's competing S1s and the S1's other records. Head = edge logit.
Batch : a set of S1 nodes + all edges of their records (2-hop), built on the GPU from CSR indexes; the loss uses
        only the batch's own S1 edges.
Fit   : two halves like the other models (A: S1 folds {1,2} -> scores {0,3,4}; B: {3,4} -> {1,2}); test = mean.

  python code/gnn/edge_gnn.py --feats c2_train_sim19 --s1 oof_xgb_c2_blend_v2.parquet --tag gnn_v2
  python code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_final_v2.parquet --tag gnn_v2 --split test
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init, gpu_mem  # noqa: E402
from common.io import CACHE, RESULTS  # noqa: E402

FEATS = ["cos_sum", "cos_e3", "cos_name", "cos_addr", "n_tset", "k_tset", "a_tset", "cnum_conflict", "cnum_c21",
         "num_conflict", "fnum_eq", "fnum_lev", "a_empty2", "legal_fam_cmp", "legal_added", "state_cmp", "k_eq",
         "tok_unm_21", "tok_unm_12", "is_domain", "nonlatin", "key_n_s1", "b_n", "a_n"]
LOGF = {"key_n_s1", "b_n", "a_n"}


class EdgeGNN(nn.Module):
    def __init__(self, f, h=96, rounds=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f, h), nn.GELU(), nn.Linear(h, h))
        self.upd = nn.ModuleList([nn.Sequential(nn.LayerNorm(5 * h), nn.Linear(5 * h, 2 * h), nn.GELU(),
                                                nn.Linear(2 * h, h)) for _ in range(rounds)])
        self.head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 1))

    @staticmethod
    def pool(h, idx, n):
        cnt = torch.zeros(n, device=h.device, dtype=h.dtype).index_add_(0, idx, torch.ones_like(idx, dtype=h.dtype))
        mean = torch.zeros(n, h.shape[1], device=h.device, dtype=h.dtype).index_add_(0, idx, h) / cnt.clamp(min=1)[:, None]
        mx = torch.full((n, h.shape[1]), -1e4, device=h.device, dtype=h.dtype).scatter_reduce(
            0, idx[:, None].expand_as(h), h, reduce="amax", include_self=True)
        return mean, mx

    def forward(self, x, ai, bi, na, nb):
        h = self.enc(x)
        for upd in self.upd:
            ma, xa = self.pool(h, ai, na)
            mb, xb = self.pool(h, bi, nb)
            h = h + upd(torch.cat([h, ma[ai], xa[ai], mb[bi], xb[bi]], 1))
        return self.head(h).squeeze(-1)


def ranges(ptr, nodes, perm=None):
    """edge ids of the given nodes from a CSR pointer (optionally through a permutation)."""
    st, en = ptr[nodes], ptr[nodes + 1]
    ln = en - st
    tot = int(ln.sum())
    off = torch.cumsum(ln, 0) - ln
    e = torch.repeat_interleave(st - off, ln, output_size=tot) + torch.arange(tot, device=ptr.device)
    return perm[e] if perm is not None else e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--s1", required=True, help="parquet under data/cache/feats with a, b, p aligned to the parts")
    ap.add_argument("--tag", default="gnn_v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_s1", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--extra", default="", help="extra edge inputs path:col under data/cache (missing -> 0 + flag)")
    ap.add_argument("--halves", default="A,B", help="test: which half models to average")
    ap.add_argument("--out_tag", default="", help="test: name of the score file (default = tag)")
    ap.add_argument("--score_only", action="store_true", help="train split: no fitting, score --score_folds with --halves")
    ap.add_argument("--score_folds", default="0")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    gdir = os.path.join(CACHE, "gnn")
    rdir = os.path.join(RESULTS, "gnn", args.tag)
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(rdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", args.feats, "part_*.parquet")))
    with Timer("load edges to the GPU"):
        P = pl.read_parquet(os.path.join(CACHE, "feats", args.s1))
        cols = ["a", "b"] + (["y", "fold"] if args.split == "train" else [])

        extras = [x.split(":") for x in args.extra.split(",") if x]
        EXT = [pl.read_parquet(os.path.join(CACHE, pth)).select(["a", "b", c]) for pth, c in extras]

        def part_x(f, off):
            d = pl.read_parquet(f, columns=cols + FEATS)
            for ex, (_, c) in zip(EXT, extras):
                d = d.join(ex, on=["a", "b"], how="left", maintain_order="left")
                d = d.with_columns(pl.col(c).is_not_null().cast(pl.Float32).alias(f"has_{c}"), pl.col(c).fill_null(0.0))
            p = P[off:off + d.height]
            assert (p["b"].to_numpy() == d["b"].to_numpy()).all() and (p["a"].to_numpy() == d["a"].to_numpy()).all()
            pp = np.clip(p["p"].to_numpy(), 1e-5, 1 - 1e-5)
            X = d.select([(pl.col(c).cast(pl.Float32).log1p() if c in LOGF else pl.col(c).cast(pl.Float32)) for c in FEATS]
                         + [pl.col(x).cast(pl.Float32) for _, c in extras for x in (c, f"has_{c}")]).to_numpy()
            return np.c_[np.log(pp / (1 - pp)), X].astype(np.float32), d.select(cols)

        # standardization statistics: from the training run's checkpoint on test, else from a sample of parts
        if args.split == "train" and not args.score_only:
            smp = np.concatenate([part_x(files[i], sum(pl.scan_parquet(files[j]).select(pl.len()).collect().item()
                                                        for j in range(i)))[0][::20] for i in range(0, len(files), 6)])
            mu, sd = smp.mean(0), np.maximum(smp.std(0), 1e-3)
        else:
            ck0 = torch.load(os.path.join(gdir, f"{args.tag}_A.pt"), map_location="cpu")
            mu, sd = ck0["mu"].numpy(), ck0["sd"].numpy()
        xs, meta, off = [], [], 0
        for f in files:
            x, m = part_x(f, off)
            off += len(x)
            xs.append(torch.from_numpy(np.clip((x - mu) / sd, -8, 8)).to(dev, torch.float16))
            meta.append(m)
        X = torch.cat(xs)
        del xs
        M = pl.concat(meta)
        mu, sd = torch.from_numpy(mu), torch.from_numpy(sd)
        a = torch.from_numpy(M["a"].to_numpy().astype(np.int64)).to(dev)
        b = torch.from_numpy(M["b"].to_numpy().astype(np.int64)).to(dev)
        E = X.shape[0]
        n1, nr = int(a.max()) + 1, int(b.max()) + 1
        perm_a = torch.argsort(a)
        a_ptr = torch.zeros(n1 + 1, dtype=torch.int64, device=dev)
        a_ptr[1:] = torch.cumsum(torch.bincount(a, minlength=n1), 0)
        b_ptr = torch.zeros(nr + 1, dtype=torch.int64, device=dev)       # parts are sorted by b
        b_ptr[1:] = torch.cumsum(torch.bincount(b, minlength=nr), 0)
        if args.split == "train":
            y = torch.from_numpy(M["y"].to_numpy().astype(np.float32)).to(dev)
            fold_e = M["fold"].to_numpy()
            s1_fold = np.full(n1, -1, np.int64)
            s1_fold[M["a"].to_numpy()] = fold_e
        print("edges", E, "S1", n1, "records", nr, gpu_mem(), flush=True)

    def subgraph(A):
        ea = ranges(a_ptr, A, perm_a)
        B = torch.unique(b[ea])
        eb = ranges(b_ptr, B)
        e = torch.unique(torch.cat([ea, eb]))
        ua, ai = torch.unique(a[e], return_inverse=True)
        ub, bi = torch.unique(b[e], return_inverse=True)
        own = torch.isin(a[e], A)
        return e, ai, bi, len(ua), len(ub), own

    deg = (a_ptr[1:] - a_ptr[:-1]).cpu().numpy()
    BIG = 5000     # S1 with more candidate records than this (a handful of very generic names) go alone

    def predict(model, s1_nodes):
        out = torch.zeros(E, device=dev)
        model.eval()
        small = s1_nodes[deg[s1_nodes] <= BIG]
        big = s1_nodes[deg[s1_nodes] > BIG]
        batches = [small[s:s + args.batch_s1 * 2] for s in range(0, len(small), args.batch_s1 * 2)] + [big[i:i + 1] for i in range(len(big))]
        with torch.no_grad():
            for nodes in batches:
                A = torch.from_numpy(nodes).to(dev)
                e, ai, bi, na, nb, own = subgraph(A)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(X[e].float(), ai, bi, na, nb).float()
                out[e[own]] = lg[own]
        return out

    logits = torch.zeros(E, device=dev)
    rep = {"args": vars(args), "edges": E}
    if args.split == "train" and args.score_only:
        present = np.unique(M["a"].to_numpy())
        sf = [int(x) for x in args.score_folds.split(",")]
        halves = args.halves.split(",")
        sc_nodes = present[np.isin(s1_fold[present], sf)]
        for name in halves:
            ck = torch.load(os.path.join(gdir, f"{args.tag}_{name}.pt"), map_location=dev)
            model = EdgeGNN(X.shape[1]).to(dev)
            model.load_state_dict(ck["state"])
            logits += predict(model, sc_nodes) / len(halves)
        keep_e = np.isin(fold_e, sf)
        from sklearn.metrics import average_precision_score
        rep["ap_scored"] = float(average_precision_score(M["y"].to_numpy()[keep_e], logits.cpu().numpy()[keep_e]))
        M = M.filter(pl.Series(keep_e))
        logits = logits[torch.from_numpy(keep_e).to(dev)]
    elif args.split == "train":
        present = np.unique(M["a"].to_numpy())
        for name, fit_f, score_f in (("A", [1, 2], [0, 3, 4]), ("B", [3, 4], [1, 2])):
            torch.manual_seed(ord(name))
            rng = np.random.default_rng(ord(name))
            model = EdgeGNN(X.shape[1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            tr_nodes = present[np.isin(s1_fold[present], fit_f) & (deg[present] <= BIG)]
            steps = args.epochs * int(np.ceil(len(tr_nodes) / args.batch_s1))
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
            with Timer(f"GNN {name}: train on {len(tr_nodes)} S1"):
                for ep in range(args.epochs):
                    model.train()
                    order = rng.permutation(tr_nodes)
                    tl, n = 0.0, 0
                    for s in range(0, len(order), args.batch_s1):
                        A = torch.from_numpy(np.sort(order[s:s + args.batch_s1])).to(dev)
                        e, ai, bi, na, nb, own = subgraph(A)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            lg = model(X[e].float(), ai, bi, na, nb).float()
                        loss = F.binary_cross_entropy_with_logits(lg[own], y[e[own]])
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        sched.step()
                        tl += loss.item() * int(own.sum())
                        n += int(own.sum())
                    print(f"    epoch {ep} loss {tl / n:.5f}", flush=True)
            torch.save({"state": model.state_dict(), "mu": mu, "sd": sd}, os.path.join(gdir, f"{args.tag}_{name}.pt"))
            with Timer(f"GNN {name}: score"):
                sc_nodes = present[np.isin(s1_fold[present], score_f)]
                lg = predict(model, sc_nodes)
                m = torch.from_numpy(np.isin(fold_e, score_f)).to(dev)
                logits[m] = lg[m]
        from sklearn.metrics import average_precision_score, roc_auc_score
        v = fold_e == 0
        yy = M["y"].to_numpy()
        lg_np = logits.cpu().numpy()
        rep.update({"val_auc_gnn": float(roc_auc_score(yy[v], lg_np[v])), "val_ap_gnn": float(average_precision_score(yy[v], lg_np[v])),
                    "val_ap_stage1": float(average_precision_score(yy[v], P["p"].to_numpy()[v]))})
    else:
        present = np.unique(M["a"].to_numpy())
        halves = args.halves.split(",")
        for name in halves:
            ck = torch.load(os.path.join(gdir, f"{args.tag}_{name}.pt"), map_location=dev)
            model = EdgeGNN(X.shape[1]).to(dev)
            model.load_state_dict(ck["state"])
            logits += predict(model, present) / len(halves)
    out = M.select(["a", "b"]).with_columns(pl.Series("gnn", logits.cpu().numpy()))
    out.write_parquet(os.path.join(gdir, f"{args.out_tag or args.tag}_{args.split}_scores.parquet"))
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(os.path.join(rdir, f"report_{args.split}{'_' + args.out_tag if args.out_tag else ''}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
