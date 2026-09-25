"""Small GPU helpers. Every job calls gpu_init() first, so it fails loudly if CUDA is not there
(we never fall back to CPU for training without noticing)."""
import os
import time

import torch

_KEEP = None


def gpu_init(require=True):
    global _KEEP
    ok = torch.cuda.is_available()
    if not ok:
        msg = f"CUDA not available (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})"
        if require:
            raise SystemExit(msg)
        print(msg, flush=True)
        return None
    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(dev)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GiB, SMs {p.multi_processor_count}, torch {torch.__version__}, "
          f"cuda {torch.version.cuda}", flush=True)
    _KEEP = torch.ones(1, device=dev)   # hold a context for the whole job
    return dev


def gpu_mem():
    if not torch.cuda.is_available():
        return "-"
    return f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} ...", flush=True)
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t
        print(f"[{time.strftime('%H:%M:%S')}] {self.label} done in {self.s:.1f}s", flush=True)
