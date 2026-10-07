"""Edge GNN as a library: the model and training loop of code/gnn/edge_gnn.py (V3), with
  * a configurable edge-feature list (e.g. without cos_e3),
  * feature overrides (e.g. test cosines computed the validation way),
  * any stage-1 table / extra edge inputs (e.g. chain-consistent scores),
  * node masks for fitting and scoring (e.g. leave-one-country-out).
edge_gnn.py itself is left untouched so V3 stays reproducible; with the default arguments this code builds the same
inputs, the same model and the same training schedule.
"""
import glob
import os

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common.io import CACHE
from gnn.edge_gnn import FEATS, LOGF, EdgeGNN, ranges
from phase2.p2lib import apply_override, cpath

BIG = 5000


class Graph:
    pass


def build_graph(feats_dir, s1, dev, extras=(), feat_list=None, override=None, mu_sd=None, with_labels=True,
                sample_every=20, sample_part_step=6):
    """feats_dir: name under feats/ (parts sorted by b, a). s1: frame a, b, p aligned row-for-row with the parts (or a
    path). extras: [(path_or_frame, column)], missing pairs -> 0 + has_<col> flag (as edge_gnn.py). override: frame
    a, b, cols replacing part columns. mu_sd: standardisation from a checkpoint (else computed on a sample)."""
    feat_list = list(feat_list or FEATS)
    files = sorted(glob.glob(os.path.join(CACHE, "feats", feats_dir, "part_*.parquet")))
    P = pl.read_parquet(cpath(s1)) if isinstance(s1, str) else s1
    cols = ["a", "b"] + (["y", "fold"] if with_labels else [])
    EXT = []
    for src, c in extras:
        e = pl.read_parquet(cpath(src)) if isinstance(src, str) else src
        EXT.append((e.select(["a", "b", c]), c))

    def part_x(f, off):
        d = pl.read_parquet(f, columns=list(dict.fromkeys(cols + feat_list)))
        if override is not None:
            d = apply_override(d, override.select(["a", "b"] + [c for c in override.columns if c in feat_list]))
        for ex, c in EXT:
            d = d.join(ex, on=["a", "b"], how="left", maintain_order="left")
            d = d.with_columns(pl.col(c).is_not_null().cast(pl.Float32).alias(f"has_{c}"), pl.col(c).fill_null(0.0))
        p = P[off:off + d.height]
        assert (p["b"].to_numpy() == d["b"].to_numpy()).all() and (p["a"].to_numpy() == d["a"].to_numpy()).all()
        pp = np.clip(p["p"].to_numpy(), 1e-5, 1 - 1e-5)
        X = d.select([(pl.col(c).cast(pl.Float32).log1p() if c in LOGF else pl.col(c).cast(pl.Float32)) for c in feat_list]
                     + [pl.col(x).cast(pl.Float32) for _, c in EXT for x in (c, f"has_{c}")]).to_numpy()
        return np.c_[np.log(pp / (1 - pp)), X].astype(np.float32), d.select(cols)

    sizes = [pl.scan_parquet(f).select(pl.len()).collect().item() for f in files]
    offs = np.r_[0, np.cumsum(sizes)]
    assert offs[-1] == P.height, (offs[-1], P.height)
    if mu_sd is None:
        smp = np.concatenate([part_x(files[i], offs[i])[0][::sample_every] for i in range(0, len(files), sample_part_step)])
        mu, sd = smp.mean(0), np.maximum(smp.std(0), 1e-3)
    else:
        mu, sd = [np.asarray(x) for x in mu_sd]
        n_in = 1 + len(feat_list) + 2 * len(EXT)
        assert len(mu) == n_in, (f"checkpoint expects {len(mu)} edge inputs, this graph has {n_in} "
                                 f"(stage-1 logit + {len(feat_list)} features + 2 per extra input): check --extra / --drop_feats")
    xs, meta = [], []
    for i, f in enumerate(files):
        x, m = part_x(f, offs[i])
        xs.append(torch.from_numpy(np.clip((x - mu) / sd, -8, 8)).to(dev, torch.float16))
        meta.append(m)
    G = Graph()
    G.dev, G.feat_list, G.extras = dev, feat_list, [c for _, c in EXT]
    G.X = torch.cat(xs)
    G.M = pl.concat(meta)
    G.mu, G.sd = torch.from_numpy(np.asarray(mu, np.float32)), torch.from_numpy(np.asarray(sd, np.float32))
    G.a = torch.from_numpy(G.M["a"].to_numpy().astype(np.int64)).to(dev)
    G.b = torch.from_numpy(G.M["b"].to_numpy().astype(np.int64)).to(dev)
    G.E = G.X.shape[0]
    G.n1, G.nr = int(G.a.max()) + 1, int(G.b.max()) + 1
    G.perm_a = torch.argsort(G.a)
    G.a_ptr = torch.zeros(G.n1 + 1, dtype=torch.int64, device=dev)
    G.a_ptr[1:] = torch.cumsum(torch.bincount(G.a, minlength=G.n1), 0)
    G.b_ptr = torch.zeros(G.nr + 1, dtype=torch.int64, device=dev)
    G.b_ptr[1:] = torch.cumsum(torch.bincount(G.b, minlength=G.nr), 0)
    G.deg = (G.a_ptr[1:] - G.a_ptr[:-1]).cpu().numpy()
    G.present = np.unique(G.M["a"].to_numpy())
    if with_labels:
        G.y = torch.from_numpy(G.M["y"].to_numpy().astype(np.float32)).to(dev)
        G.fold_e = G.M["fold"].to_numpy()
    print("graph: edges", G.E, "S1", G.n1, "records", G.nr, "inputs", G.X.shape[1], flush=True)
    return G


def _subgraph(G, A):
    ea = ranges(G.a_ptr, A, G.perm_a)
    B = torch.unique(G.b[ea])
    eb = ranges(G.b_ptr, B)
    e = torch.unique(torch.cat([ea, eb]))
    ua, ai = torch.unique(G.a[e], return_inverse=True)
    ub, bi = torch.unique(G.b[e], return_inverse=True)
    own = torch.isin(G.a[e], A)
    return e, ai, bi, len(ua), len(ub), own


def _ac(dev):
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def train(G, fit_nodes, epochs=4, batch_s1=1500, lr=2e-3, seed=0):
    dev = G.dev
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EdgeGNN(G.X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tr = fit_nodes[G.deg[fit_nodes] <= BIG]
    steps = epochs * int(np.ceil(len(tr) / batch_s1))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.05)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        tl, n = 0.0, 0
        for s in range(0, len(order), batch_s1):
            A = torch.from_numpy(np.sort(order[s:s + batch_s1])).to(dev)
            e, ai, bi, na, nb, own = _subgraph(G, A)
            with _ac(dev):
                lg = model(G.X[e].float(), ai, bi, na, nb).float()
            loss = F.binary_cross_entropy_with_logits(lg[own], G.y[e[own]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * int(own.sum())
            n += int(own.sum())
        print(f"    epoch {ep} loss {tl / max(n, 1):.5f}", flush=True)
    return model


def predict(G, model, nodes, batch_s1=1500):
    out = torch.zeros(G.E, device=G.dev)
    model.eval()
    small, big = nodes[G.deg[nodes] <= BIG], nodes[G.deg[nodes] > BIG]
    batches = [small[s:s + batch_s1 * 2] for s in range(0, len(small), batch_s1 * 2)] + [big[i:i + 1] for i in range(len(big))]
    with torch.no_grad():
        for nd in batches:
            A = torch.from_numpy(nd).to(G.dev)
            e, ai, bi, na, nb, own = _subgraph(G, A)
            with _ac(G.dev):
                lg = model(G.X[e].float(), ai, bi, na, nb).float()
            out[e[own]] = lg[own]
    return out


def save(G, model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": G.mu, "sd": G.sd, "feats": G.feat_list, "extras": G.extras}, path)


def load(path, dev):
    ck = torch.load(path, map_location=dev)
    return ck


def model_from(ck, n_in, dev):
    m = EdgeGNN(n_in).to(dev)
    m.load_state_dict(ck["state"])
    return m


def s1_folds(G):
    f = np.full(G.n1, -1, np.int64)
    f[G.M["a"].to_numpy()] = G.fold_e
    return f


HALVES = (("A", [1, 2], [0, 3, 4]), ("B", [3, 4], [1, 2]))


def crossfit(G, ckpt_prefix, fit_mask=None, epochs=4, batch_s1=1500, lr=2e-3):
    """two halves as edge_gnn.py; fit_mask: optional bool over S1 rows restricting the fitting nodes (LOCO).
    Every present S1 is scored by the half that did not fit its fold. Returns logits (E,) as numpy."""
    sf = s1_folds(G)
    logits = torch.zeros(G.E, device=G.dev)
    for name, fit_f, score_f in HALVES:
        nodes = G.present[np.isin(sf[G.present], fit_f)]
        if fit_mask is not None:
            nodes = nodes[fit_mask[nodes]]
        model = train(G, nodes, epochs, batch_s1, lr, seed=ord(name))
        save(G, model, f"{ckpt_prefix}_{name}.pt")
        sc_nodes = G.present[np.isin(sf[G.present], score_f)]
        lg = predict(G, model, sc_nodes, batch_s1)
        m = torch.from_numpy(np.isin(G.fold_e, score_f)).to(G.dev)
        logits[m] = lg[m]
        del model
    return logits.cpu().numpy()


def score_with(G, ckpt_paths, nodes=None, batch_s1=1500):
    """mean logit of the given checkpoints over the nodes (default: every S1 present)."""
    nodes = G.present if nodes is None else nodes
    out = torch.zeros(G.E, device=G.dev)
    for p in ckpt_paths:
        ck = load(p, G.dev)
        out += predict(G, model_from(ck, G.X.shape[1], G.dev), nodes, batch_s1) / len(ckpt_paths)
    return out.cpu().numpy()
