"""Multilingual cross-encoder for the uncertain pairs (pretrained XLM-RoBERTa base, MIT licence, 278M params).

Input : "<s> S1 name | address </s></s> record name | address </s>" on the RAW text (native scripts and accents kept:
        the pretrained model has seen Hindi, Tamil, French ...), 128 tokens max.
Head  : linear on the first token, BCE loss, AdamW 2e-5, linear warmup/decay, bf16 autocast on the GPU.
Fit   : two halves (A on folds {1,2} scores {0,3,4}; B on {3,4} scores {1,2}); test = mean of A and B.

  python code/neural_reranker/ml_cross_encoder.py --name hardv2 --split train
  python code/neural_reranker/ml_cross_encoder.py --name hardv2 --split test --score_only
V6 adds --model (any MIT / Apache-2.0 encoder on the Hugging Face hub, e.g. BAAI/bge-reranker-v2-m3, 568M, Apache-2.0,
a multilingual cross-encoder pretrained for relevance ranking: its own classification head is kept and fine-tuned) and
--col (name of the score column). Defaults reproduce V3 exactly.
  python code/neural_reranker/ml_cross_encoder.py --name v6band --split train --model BAAI/bge-reranker-v2-m3 --col bgexenc
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

MODEL = "FacebookAI/xlm-roberta-base"
MAXLEN = 128
LR, BS = 0.0, 0                        # set by --lr / --bs (0 = the V3 defaults 2e-5 / 64)


class PairModel(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
        arch = AutoConfig.from_pretrained(MODEL).architectures or []
        self.seqcls = any(a.endswith("ForSequenceClassification") for a in arch)
        if self.seqcls:                     # a pretrained cross-encoder (reranker): keep its relevance head
            self.enc = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)
        else:
            self.enc = AutoModel.from_pretrained(MODEL)
            self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, mask):
        if self.seqcls:
            return self.enc(input_ids=ids, attention_mask=mask).logits[:, 0]
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


def tokenize(t1, t2, bs=100_000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = np.full((len(t1), MAXLEN), tok.pad_token_id, np.int32)
    for s in range(0, len(t1), bs):
        x = tok(t1[s:s + bs], t2[s:s + bs], truncation="longest_first", max_length=MAXLEN, padding="max_length")
        ids[s:s + bs] = np.asarray(x["input_ids"], np.int32)
    return ids, tok.pad_token_id


def train(ids, pad, y, dev, epochs, bs=64, lr=2e-5, seed=0):
    bs, lr = BS or bs, LR or lr
    torch.manual_seed(seed)
    model = PairModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(y)
    steps = epochs * math.ceil(n / bs)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    I = torch.from_numpy(ids).to(dev)
    Y = torch.from_numpy(y.astype(np.float32)).to(dev)
    rng = np.random.default_rng(seed)
    step = 0
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        tl, t0 = 0.0, time.time()
        model.train()
        for s in range(0, n, bs):
            j = perm[s:s + bs]
            x = I[j].long()
            L = int((x != pad).sum(1).max())          # trim padding of this batch
            x = x[:, :L]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x, (x != pad).long())
            loss = F.binary_cross_entropy_with_logits(logit.float(), Y[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            tl += loss.item() * len(j)
            if step % 2000 == 0:
                print(f"    step {step}/{steps} loss {tl / (s + len(j)):.4f} {time.time() - t0:.0f}s", flush=True)
        print(f"    epoch {ep} loss {tl / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def score(model, ids, pad, dev, bs=512):
    model.eval()
    out = np.zeros(len(ids), np.float32)
    # sort by length so batches carry little padding
    lens = (ids != pad).sum(1)
    order = np.argsort(lens)
    for s in range(0, len(ids), bs):
        o = order[s:s + bs]
        x = torch.from_numpy(ids[o]).to(dev).long()
        L = int((x != pad).sum(1).max())
        x = x[:, :L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[o] = model(x, (x != pad).long()).float().cpu().numpy()
    return out


def main():
    global MODEL, LR, BS
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hardv2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=400_000)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--model", default=MODEL, help="Hugging Face model id (MIT / Apache-2.0 licence, <= 8B parameters)")
    ap.add_argument("--col", default="mlxenc", help="name of the score column in the output")
    ap.add_argument("--lr", type=float, default=0.0, help="0 = 2e-5")
    ap.add_argument("--bs", type=int, default=0, help="0 = 64")
    args = ap.parse_args()
    MODEL, LR, BS = args.model, args.lr, args.bs
    dev = gpu_init()
    t0 = time.time()
    xdir = os.path.join(CACHE, "xenc")
    rdir = os.path.join(RESULTS, "neural_reranker", f"ml_{args.name}")
    os.makedirs(rdir, exist_ok=True)
    H = pl.read_parquet(os.path.join(xdir, f"{args.name}_{args.split}.parquet"))
    with Timer(f"tokenize {H.height} pairs"):
        ids, pad = tokenize(H["t1"].to_list(), H["t2"].to_list())
    rep = {"args": vars(args), "pairs": H.height, "model": MODEL}
    xenc = np.zeros(H.height, np.float32)
    if args.split == "train" and not args.score_only:
        fold, y = H["fold"].to_numpy(), H["y"].to_numpy()
        for name, fit_f, score_f in (("A", [1, 2], [0, 3, 4]), ("B", [3, 4], [1, 2])):
            tr = np.where(np.isin(fold, fit_f))[0]
            if len(tr) > args.max_train:
                tr = np.random.default_rng(0).choice(tr, args.max_train, replace=False)
            with Timer(f"model {name}: fine-tune on {len(tr)} pairs"):
                m = train(ids[tr], pad, y[tr], dev, args.epochs, seed=ord(name))
                torch.save(m.state_dict(), os.path.join(xdir, f"ml_{args.name}_model_{name}.pt"))
            te = np.where(np.isin(fold, score_f))[0]
            with Timer(f"model {name}: score {len(te)} pairs"):
                xenc[te] = score(m, ids[te], pad, dev)
            del m
            torch.cuda.empty_cache()
        from sklearn.metrics import average_precision_score, roc_auc_score
        v = fold == 0
        p1 = H["p"].to_numpy()
        if 0 < y[v].sum() < v.sum():
            rep.update({"val_auc_mlxenc": float(roc_auc_score(y[v], xenc[v])), "val_auc_stage1": float(roc_auc_score(y[v], p1[v])),
                        "val_ap_mlxenc": float(average_precision_score(y[v], xenc[v])),
                        "val_ap_stage1": float(average_precision_score(y[v], p1[v]))})
    else:
        for name in ("A", "B"):
            m = PairModel().to(dev)
            m.load_state_dict(torch.load(os.path.join(xdir, f"ml_{args.name}_model_{name}.pt"), map_location=dev))
            xenc += score(m, ids, pad, dev) / 2
            del m
    H.select(["a", "b"]).with_columns(pl.Series(args.col, xenc)).write_parquet(
        os.path.join(xdir, f"ml_{args.name}_{args.split}_scores.parquet"))
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(os.path.join(rdir, f"report_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
