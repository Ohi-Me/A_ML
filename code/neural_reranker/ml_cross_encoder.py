"""Multilingual cross-encoder for the uncertain pairs (pretrained XLM-RoBERTa base, MIT licence, 278M params).

Input : "<s> S1 name | address </s></s> record name | address </s>" on the RAW text (native scripts and accents kept:
        the pretrained model has seen Hindi, Tamil, French ...), 128 tokens max.
Head  : linear on the first token, BCE loss, AdamW 2e-5, linear warmup/decay, bf16 autocast on the GPU.
Fit   : two halves (A on folds {1,2} scores {0,3,4}; B on {3,4} scores {1,2}); test = mean of A and B.

  python code/neural_reranker/ml_cross_encoder.py --name hardv2 --split train
  python code/neural_reranker/ml_cross_encoder.py --name hardv2 --split test --score_only
V6 adds --model (any MIT / Apache-2.0 model on the Hugging Face hub below 8B parameters, e.g. BAAI/bge-reranker-v2-m3,
568M, Apache-2.0: its own relevance head is kept and fine-tuned) and --col (name of the score column).
V7 adds
  --out_name      prefix of every output (default ml_<name>, so V3 / V6 commands write the same files as before)
  --models_from   score with the A / B checkpoints of an earlier run (e.g. ml_hardv2): on train each fold is scored by
                  the half that did NOT see it (A: folds 0,3,4; B: folds 1,2), so the scores stay out-of-fold
  decoder-only rerankers (architecture *ForCausalLM, e.g. Qwen/Qwen3-Reranker-0.6B or -4B, Apache-2.0): the official
                  yes/no prompt; score = logit(yes) - logit(no) at the last token, fine-tuned with BCE through a
                  trainable copy of the (yes - no) output row; embeddings frozen
  --train_top K   train only the last K transformer layers (+ head): parameter-efficient fine-tuning for large models
  python code/neural_reranker/ml_cross_encoder.py --name v7band --split train --score_only --models_from ml_hardv2 --out_name ml_v7xlmr
  python code/neural_reranker/ml_cross_encoder.py --name v7band --split train --model Qwen/Qwen3-Reranker-0.6B --col qwxenc --out_name ml_v7qwen
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
LR, BS, TOP = 0.0, 0, 0                # set by --lr / --bs / --train_top (0 = the V3 defaults 2e-5 / 64 / all layers)
Q_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
            "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
Q_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
Q_INSTRUCT = ("Do the Query and the Document describe the same business? Names and addresses may contain typos, "
              "abbreviations, another script, a different legal form or a changed house number.")
Q_FIELD = 64                           # tokens per record in the causal prompt


def arch_kind(model_id):
    from transformers import AutoConfig
    arch = AutoConfig.from_pretrained(model_id).architectures or []
    if any(a.endswith("ForCausalLM") for a in arch):
        return "causal"
    if any(a.endswith("ForSequenceClassification") for a in arch):
        return "seqcls"
    return "enc"


def layer_list(m):
    """the transformer block list of an encoder / decoder (for --train_top)."""
    for path in ("layers", "encoder.layer", "roberta.encoder.layer", "model.layers"):
        o = m
        try:
            for p in path.split("."):
                o = getattr(o, p)
            return o
        except AttributeError:
            continue
    return None


class PairModel(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        self.kind = arch_kind(MODEL)
        self.seqcls = self.kind == "seqcls"
        if self.kind == "causal":
            lm = AutoModelForCausalLM.from_pretrained(MODEL).float()
            tok = AutoTokenizer.from_pretrained(MODEL)
            W = lm.get_output_embeddings().weight.detach()
            yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
            self.w = nn.Parameter((W[yes] - W[no]).clone().float())
            self.b = nn.Parameter(torch.zeros(1))
            self.enc = lm.base_model
            del lm
            self.enc.get_input_embeddings().weight.requires_grad_(False)
        elif self.seqcls:                   # a pretrained cross-encoder (reranker): keep its relevance head
            self.enc = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).float()
        else:
            self.enc = AutoModel.from_pretrained(MODEL).float()
            self.head = nn.Linear(self.enc.config.hidden_size, 1)
        if TOP > 0:
            layers = layer_list(self.enc)
            assert layers is not None, "--train_top: transformer layers not found"
            for p in self.enc.parameters():
                p.requires_grad_(False)
            for blk in list(layers)[-TOP:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            for name, mod in self.enc.named_modules():       # final norm and the classification head stay trainable
                if name.endswith(("norm", "classifier", "pooler")) and "layers" not in name and "layer." not in name:
                    for p in mod.parameters():
                        p.requires_grad_(True)

    def forward(self, ids, mask):
        if self.kind == "causal":
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            last = mask.sum(1) - 1
            return h[torch.arange(len(h), device=h.device), last].float() @ self.w + self.b
        if self.seqcls:
            return self.enc(input_ids=ids, attention_mask=mask).logits[:, 0]
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(h).squeeze(-1)


def tokenize(t1, t2, bs=100_000):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if arch_kind(MODEL) == "causal":        # right-padded yes/no prompt, record texts truncated to Q_FIELD tokens each
        pre = tok.encode(Q_PREFIX + f"<Instruct>: {Q_INSTRUCT}\n<Query>: ", add_special_tokens=False)
        mid = tok.encode("\n<Document>: ", add_special_tokens=False)
        suf = tok.encode(Q_SUFFIX, add_special_tokens=False)
        width = len(pre) + len(mid) + len(suf) + 2 * Q_FIELD
        ids = np.full((len(t1), width), pad, np.int32)
        for s in range(0, len(t1), bs):
            q = tok(t1[s:s + bs], add_special_tokens=False, truncation=True, max_length=Q_FIELD)["input_ids"]
            d = tok(t2[s:s + bs], add_special_tokens=False, truncation=True, max_length=Q_FIELD)["input_ids"]
            for i, (x, y) in enumerate(zip(q, d)):
                row = pre + x + mid + y + suf
                ids[s + i, :len(row)] = row
        return ids, pad
    ids = np.full((len(t1), MAXLEN), pad, np.int32)
    for s in range(0, len(t1), bs):
        x = tok(t1[s:s + bs], t2[s:s + bs], truncation="longest_first", max_length=MAXLEN, padding="max_length")
        ids[s:s + bs] = np.asarray(x["input_ids"], np.int32)
    return ids, pad


def train(ids, pad, y, dev, epochs, bs=64, lr=2e-5, seed=0):
    bs, lr = BS or bs, LR or lr
    torch.manual_seed(seed)
    model = PairModel().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"    trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
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
            torch.nn.utils.clip_grad_norm_(params, 1.0)
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


def save(model, path):
    """causal models: only the trained tensors (the frozen base is reloaded from the hub); others: everything."""
    sd = model.state_dict()
    if getattr(model, "kind", "") == "causal":
        train_names = {n for n, p in model.named_parameters() if p.requires_grad}
        sd = {k: v for k, v in sd.items() if k in train_names}
    torch.save(sd, path)


def load(path, dev):
    m = PairModel().to(dev)
    missing, unexpected = m.load_state_dict(torch.load(path, map_location=dev), strict=False)
    if getattr(m, "kind", "") != "causal":
        assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m


def main():
    global MODEL, LR, BS, TOP
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hardv2", help="pair file xenc/<name>_<split>.parquet (prep_pairs.py)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_train", type=int, default=400_000)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--model", default=MODEL, help="Hugging Face model id (MIT / Apache-2.0 licence, < 8B parameters)")
    ap.add_argument("--col", default="mlxenc", help="name of the score column in the output")
    ap.add_argument("--lr", type=float, default=0.0, help="0 = 2e-5")
    ap.add_argument("--bs", type=int, default=0, help="0 = 64")
    ap.add_argument("--train_top", type=int, default=0, help="train only the last K transformer layers (0 = all)")
    ap.add_argument("--out_name", default="", help="prefix of the outputs (default ml_<name>)")
    ap.add_argument("--models_from", default="", help="out_name of an earlier run whose A/B checkpoints score this file")
    args = ap.parse_args()
    out = args.out_name or f"ml_{args.name}"
    src = args.models_from or out
    MODEL, LR, BS, TOP = args.model, args.lr, args.bs, args.train_top
    if args.models_from:                    # the checkpoints' own model id
        rp = os.path.join(RESULTS, "neural_reranker", src, "report_train.json")
        if os.path.exists(rp):
            MODEL = json.load(open(rp)).get("model", MODEL)
        args.score_only = True
    dev = gpu_init()
    t0 = time.time()
    xdir = os.path.join(CACHE, "xenc")
    rdir = os.path.join(RESULTS, "neural_reranker", out)
    os.makedirs(rdir, exist_ok=True)
    H = pl.read_parquet(os.path.join(xdir, f"{args.name}_{args.split}.parquet"))
    with Timer(f"tokenize {H.height} pairs"):
        ids, pad = tokenize(H["t1"].to_list(), H["t2"].to_list())
    rep = {"args": vars(args), "pairs": H.height, "model": MODEL, "checkpoints": src}
    xenc = np.zeros(H.height, np.float32)
    halves = (("A", [1, 2], [0, 3, 4]), ("B", [3, 4], [1, 2]))
    if args.split == "train" and not args.score_only:
        fold, y = H["fold"].to_numpy(), H["y"].to_numpy()
        for name, fit_f, score_f in halves:
            tr = np.where(np.isin(fold, fit_f))[0]
            if len(tr) > args.max_train:
                tr = np.random.default_rng(0).choice(tr, args.max_train, replace=False)
            with Timer(f"model {name}: fine-tune on {len(tr)} pairs"):
                m = train(ids[tr], pad, y[tr], dev, args.epochs, seed=ord(name))
                save(m, os.path.join(xdir, f"{out}_model_{name}.pt"))
            te = np.where(np.isin(fold, score_f))[0]
            with Timer(f"model {name}: score {len(te)} pairs"):
                xenc[te] = score(m, ids[te], pad, dev)
            del m
            torch.cuda.empty_cache()
    elif args.split == "train":             # out-of-fold scoring with existing checkpoints
        fold = H["fold"].to_numpy()
        for name, _, score_f in halves:
            te = np.where(np.isin(fold, score_f))[0]
            m = load(os.path.join(xdir, f"{src}_model_{name}.pt"), dev)
            with Timer(f"checkpoint {src} {name}: score {len(te)} out-of-fold pairs"):
                xenc[te] = score(m, ids[te], pad, dev)
            del m
            torch.cuda.empty_cache()
    else:
        for name in ("A", "B"):
            m = load(os.path.join(xdir, f"{src}_model_{name}.pt"), dev)
            xenc += score(m, ids, pad, dev) / 2
            del m
            torch.cuda.empty_cache()
    if args.split == "train":
        from sklearn.metrics import average_precision_score, roc_auc_score
        fold, y = H["fold"].to_numpy(), H["y"].to_numpy()
        v = fold == 0
        p1 = H["p"].to_numpy()
        if 0 < y[v].sum() < v.sum():
            rep.update({"val_auc_mlxenc": float(roc_auc_score(y[v], xenc[v])), "val_auc_stage1": float(roc_auc_score(y[v], p1[v])),
                        "val_ap_mlxenc": float(average_precision_score(y[v], xenc[v])),
                        "val_ap_stage1": float(average_precision_score(y[v], p1[v]))})
    H.select(["a", "b"]).with_columns(pl.Series(args.col, xenc)).write_parquet(
        os.path.join(xdir, f"{out}_{args.split}_scores.parquet"))
    rep["runtime_s"] = time.time() - t0
    rep["gpu"] = gpu_mem()
    print(json.dumps(rep), flush=True)
    json.dump(rep, open(os.path.join(rdir, f"report_{args.split}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
