"""Neural reranker, step 2: character-level cross-encoder trained from scratch on the uncertain pairs.

Input  : [CLS] S1 "name | address" [SEP] record "name | address"  (ascii bytes, <= 96 chars each)
Model  : 4-layer transformer encoder (d 256, 8 heads), learned positions + segment embeddings, CLS -> logit
Fitting: two halves like the pair models: A on folds {1,2} scores {0,3,4}, B on {3,4} scores {1,2};
         test pairs get the mean of A and B. bf16 autocast on the GPU.
Output : data/cache/xenc/<name>_<split>_scores.parquet with a, b, xenc (logit).

  python code/neural_reranker/cross_encoder.py --name hard --split train
  python code/neural_reranker/cross_encoder.py --name hard --split test --score_only
"""
import argparse
import json
import math
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

L1 = 96
SEQ = 2 * L1 + 2


def encode(t1, t2):
    """list of str -> uint8 token ids (0 pad, 1 CLS, 2 SEP, bytes shifted by 3) and segment ids."""
    n = len(t1)
    ids = np.zeros((n, SEQ), np.uint8)
    seg = np.zeros((n, SEQ), np.uint8)
    ids[:, 0] = 1
    for i, (x, y) in enumerate(zip(t1, t2)):
        bx = np.frombuffer(x.encode("ascii", "ignore")[:L1], np.uint8)
        by = np.frombuffer(y.encode("ascii", "ignore")[:L1], np.uint8)
        ids[i, 1:1 + len(bx)] = np.minimum(bx, 124) + 3
        ids[i, 1 + len(bx)] = 2
        s = 2 + len(bx)
        ids[i, s:s + len(by)] = np.minimum(by, 124) + 3
        seg[i, s:s + len(by)] = 1
    return ids, seg


class CrossEncoder(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, ff=1024):
        super().__init__()
        self.tok = nn.Embedding(128, d)
        self.pos = nn.Embedding(SEQ, d)
        self.seg = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, ids, seg):
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        x = self.tok(ids) + self.pos(pos) + self.seg(seg)
        x = self.enc(x, src_key_padding_mask=(ids == 0))
        return self.head(self.norm(x[:, 0])).squeeze(-1)


def train(ids, seg, y, dev, epochs, bs=512, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    model = CrossEncoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(y)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    I = torch.from_numpy(ids).to(dev)
    S = torch.from_numpy(seg).to(dev)
    Y = torch.from_numpy(y.astype(np.float32)).to(dev)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        tl, t0 = 0.0, time.time()
        model.train()
        for s in range(0, n, bs):
            j = perm[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(I[j].long(), S[j].long())
            loss = F.binary_cross_entropy_with_logits(logit.float(), Y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(j)
        print(f"    epoch {ep} loss {tl / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def score(model, ids, seg, dev, bs=4096):
    model.eval()
    out = np.zeros(len(ids), np.float32)
    for s in range(0, len(ids), bs):
        i = torch.from_numpy(ids[s:s + bs]).to(dev).long()
        g = torch.from_numpy(seg[s:s + bs]).to(dev).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[s:s + bs] = model(i, g).float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hard")
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_train", type=int, default=1_500_000)
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()
    dev = gpu_init()
    t0 = time.time()
    xdir = os.path.join(CACHE, "xenc")
    rdir = os.path.join(RESULTS, "neural_reranker", args.name)
    os.makedirs(rdir, exist_ok=True)
    H = pl.read_parquet(os.path.join(xdir, f"{args.name}_{args.split}.parquet"))
    with Timer(f"encode {H.height} pairs"):
        ids, seg = encode(H["t1"].to_list(), H["t2"].to_list())
    rep = {"args": vars(args), "pairs": H.height}
    if args.split == "train" and not args.score_only:
        fold, y = H["fold"].to_numpy(), H["y"].to_numpy()
        xenc = np.zeros(H.height, np.float32)
        for name, fit_f, score_f in (("A", [1, 2], [0, 3, 4]), ("B", [3, 4], [1, 2])):
            tr = np.where(np.isin(fold, fit_f))[0]
            if len(tr) > args.max_train:
                tr = np.random.default_rng(0).choice(tr, args.max_train, replace=False)
            with Timer(f"model {name}: train on {len(tr)} pairs"):
                m = train(ids[tr], seg[tr], y[tr], dev, args.epochs, seed=ord(name))
                torch.save(m.state_dict(), os.path.join(xdir, f"{args.name}_model_{name}.pt"))
            te = np.where(np.isin(fold, score_f))[0]
            with Timer(f"model {name}: score {len(te)} pairs"):
                xenc[te] = score(m, ids[te], seg[te], dev)
            del m
            torch.cuda.empty_cache()
        from sklearn.metrics import roc_auc_score, average_precision_score
        v = fold == 0
        p1 = H["p"].to_numpy()
        rep["val_auc_xenc"] = float(roc_auc_score(y[v], xenc[v]))
        rep["val_auc_stage1"] = float(roc_auc_score(y[v], p1[v]))
        rep["val_ap_xenc"] = float(average_precision_score(y[v], xenc[v]))
        rep["val_ap_stage1"] = float(average_precision_score(y[v], p1[v]))
    else:
        xenc = np.zeros(H.height, np.float32)
        for name in ("A", "B"):
            m = CrossEncoder().to(dev)
            m.load_state_dict(torch.load(os.path.join(xdir, f"{args.name}_model_{name}.pt"), map_location=dev))
            xenc += score(m, ids, seg, dev) / 2
    H.select(["a", "b"]).with_columns(pl.Series("xenc", xenc)).write_parquet(
        os.path.join(xdir, f"{args.name}_{args.split}_scores.parquet"))
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(os.path.join(rdir, f"report_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
