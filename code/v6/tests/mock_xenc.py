"""Smoke-test stand-in for code/neural_reranker/ml_cross_encoder.py: same arguments, same files and columns, but the
Hugging Face model / tokenizer are replaced by a trivial model (no download, CPU). Numbers mean nothing."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import neural_reranker.ml_cross_encoder as M  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


M.PairModel = Tiny
M.tokenize = lambda t1, t2, bs=0: (np.array([[len(a) % 7, len(b) % 5] for a, b in zip(t1, t2)], np.int32), 0)
M.train = lambda ids, pad, y, dev, epochs, bs=64, lr=2e-5, seed=0: Tiny()
M.score = lambda m, ids, pad, dev, bs=0: np.random.default_rng(len(ids)).normal(0, 1, len(ids)).astype(np.float32)
sys.argv = ["ml_cross_encoder.py"] + sys.argv[1:]
M.main()
