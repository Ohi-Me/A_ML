"""Build a tiny synthetic stand-in for data/cache_v2 + results/ with the real schemas, so every phase-2 script can be
smoke-tested end to end on a CPU (ER_ALLOW_CPU=1). Numbers produced from it mean nothing.

  ER_ROOT=/tmp/synth ER_CACHE=/tmp/synth/data/cache_v2 ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH=code \
      python code/phase2/tests/make_synth.py
"""
import json
import os
import sys
import zlib

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common.io import CACHE, RAW_CACHE, RESULTS  # noqa: E402
from common.stage2_feats import context_features, sibling_features  # noqa: E402
from embeddings.ngram_biencoder import GCSR, Encoder, encode_rows  # noqa: E402
from phase2 import gnn_lib as GL  # noqa: E402
from phase2.p2lib import COS_CTX, ctx_features  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
FEATS = json.load(open(os.path.join(REPO, "results", "xgboost", "xgb_c2_sim19_v2", "report.json")))["features"]
DROP_EMB = ["cos_e3", "b_gap_cos_e3", "b_rank_cos_e3", "a_gap_cos_e3", "a_rank_cos_e3", "b_gap2_cos_e3", "r_e3_emb"]
PARAMS = {"objective": "binary:logistic", "eval_metric": ["logloss"], "tree_method": "hist", "device": "cpu",
          "max_depth": 4, "eta": 0.2, "max_bin": 256}
rng = np.random.default_rng(0)
dev = torch.device("cpu")
HN, HA = 300, 200


def fold_of(x):
    return zlib.crc32(x.encode()) % 5


def make_split(split, n1, countries):
    s1_id = [f"S1-{split}{i}" for i in range(n1)]
    c1 = rng.choice(countries, n1)
    k = np.where(rng.random(n1) < 0.06, 0, np.clip(rng.poisson(3, n1), 1, 8))
    rec_s1 = np.repeat(np.arange(n1), k)                    # true S1 of each matched record
    nd = int(len(rec_s1) * 0.4)
    rec_s1 = np.r_[rec_s1, np.full(nd, -1)]
    rc = np.where(rec_s1 >= 0, c1[np.maximum(rec_s1, 0)], rng.choice(countries, len(rec_s1)))
    nr = len(rec_s1)
    rid = [f"S{2 if i < nr // 2 else 3}-{split}{i}" for i in range(nr)]   # S2 block then S3 block (as ids() concatenates)
    # candidates: true S1 (97%) + 7 random S1 of the same country
    by_c = {c: np.where(c1 == c)[0] for c in countries}
    A, B = [], []
    for b in range(nr):
        pool = by_c[rc[b]]
        cand = set(rng.choice(pool, min(7, len(pool)), replace=False).tolist())
        if rec_s1[b] >= 0 and rng.random() < 0.97:
            cand.add(int(rec_s1[b]))
        A += list(cand)
        B += [b] * len(cand)
    C = pl.DataFrame({"a": np.array(A, np.int32), "b": np.array(B, np.int32)}).sort(["b", "a"])
    y = (rec_s1[C["b"].to_numpy()] == C["a"].to_numpy()).astype(np.int8)
    return s1_id, c1, rid, rc, rec_s1, C, y


def features(C, y, cos_e3=None):
    n = C.height
    s = y * 1.6 + rng.normal(0, 1, n)
    d = {}
    for f in FEATS:
        if f.startswith(("r_", "b_rank", "a_rank")):
            d[f] = np.clip((3 - s + rng.normal(0, 1, n)).round(), 0, 20).astype(np.int8 if f.startswith("r_") else np.int32)
        elif f in ("b_n", "a_n", "key_n_s1", "key_n_r"):
            d[f] = rng.integers(1, 30, n).astype(np.int32)
        else:
            d[f] = (0.3 * s + rng.normal(0, 0.5, n)).astype(np.float32)
    F = C.with_columns([pl.Series(k, v) for k, v in d.items()])
    if cos_e3 is not None:
        F = F.drop(COS_CTX).with_columns(pl.Series("cos_e3", cos_e3.astype(np.float32)))
        F = ctx_features(F, "cos_e3")
    return F


def write_parts(F, d, n_parts=3):
    os.makedirs(d, exist_ok=True)
    step = int(np.ceil(F.height / n_parts))
    for i in range(n_parts):
        F[i * step:(i + 1) * step].write_parquet(os.path.join(d, f"part_{i:03d}.parquet"))


def save_sources(split, s1_id, c1, rid, rc):
    os.makedirs(RAW_CACHE, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    words = np.array(["acme", "global", "traders", "pvt", "ltd", "inc", "sharma", "boulangerie", "sas", "llc"])
    for k, ids_, cs in ((1, s1_id, c1), (2, [r for r in rid if r.startswith("S2")], None), (3, [r for r in rid if r.startswith("S3")], None)):
        if cs is None:
            m = np.array([r.startswith(f"S{k}") for r in rid])
            cs = rc[m]
        nm = [" ".join(rng.choice(words, 3)) for _ in ids_]
        ad = [f"{rng.integers(1, 999)} main road" for _ in ids_]
        pl.DataFrame({"entity_id": ids_, "business_name": nm, "business_address": ad, "country": cs}).write_parquet(
            os.path.join(RAW_CACHE, f"{split}_source{k}.parquet"))
        pl.DataFrame({"id": ids_, "country": cs, "state": ["x"] * len(ids_)}).write_parquet(
            os.path.join(CACHE, f"norm_v2_{split}_s{k}.parquet"))


def fit(F, feats, folds, rounds=30):
    d = F.filter(pl.col("fold").is_in(folds))
    return xgb.train(PARAMS, xgb.DMatrix(d.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy(), label=d["y"].to_numpy()), rounds)


def pred(m, F, feats):
    return m.predict(xgb.DMatrix(F.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()))


def oof(F, mA, mB, feats):
    fo = F["fold"].to_numpy()
    p = np.where(np.isin(fo, [0, 3, 4]), pred(mA, F, feats), pred(mB, F, feats)).astype(np.float32)
    return F.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", p))


def main():
    for d in ("feats", "gnn", "xenc", "cands", "emb/models", "emb/e3_test", "emb/e2f_test", "blocking/b1_test"):
        os.makedirs(os.path.join(CACHE, d), exist_ok=True)
    # ---------------- train split
    s1, c1, rid, rc, rec_s1, C, y = make_split("tr", 1500, np.array(["US", "India"]))
    save_sources("train", s1, c1, rid, rc)
    gt = pl.DataFrame({"s1": [s1[a] for a in rec_s1 if a >= 0], "rid": [r for r, a in zip(rid, rec_s1) if a >= 0]})
    gt.write_parquet(os.path.join(RAW_CACHE, "train_gt_pairs.parquet"))
    gt.group_by("s1").agg(pl.col("rid").alias("matches")).write_parquet(os.path.join(RAW_CACHE, "train_gt.parquet"))
    fold1 = np.array([fold_of(x) for x in s1], np.int8)
    drop = np.array([(zlib.crc32((x + "|drop").encode()) % 10000) < 1900 for x in s1])
    np.save(os.path.join(CACHE, "drop_19.npy"), drop)
    F = features(C, y).with_columns(pl.Series("fold", fold1[C["a"].to_numpy()]), pl.Series("y", y),
                                    pl.Series("b_has_match", (rec_s1[C["b"].to_numpy()] >= 0).astype(np.int8)))
    F = F.filter(pl.Series(~drop[F["a"].to_numpy()]))
    write_parts(F, os.path.join(CACHE, "feats", "c2_train_sim19"))
    # stage 1 full / no-embedding, blend
    noemb = [f for f in FEATS if f not in DROP_EMB]
    mods = {}
    for tag, feats in (("xgb_c2_sim19_v2", FEATS), ("xgb_c2_sim19_noemb_v2", noemb)):
        rd = os.path.join(RESULTS, "xgboost", tag)
        os.makedirs(rd, exist_ok=True)
        mA, mB = fit(F, feats, [1, 2]), fit(F, feats, [3, 4])
        mA.save_model(os.path.join(rd, "model_A.json"))
        mB.save_model(os.path.join(rd, "model_B.json"))
        json.dump({"features": feats, "params": PARAMS, "rounds": 30}, open(os.path.join(rd, "report.json"), "w"))
        mods[tag] = (mA, mB, feats)
        os.makedirs(os.path.join(RESULTS, "final", "final_v2"), exist_ok=True)
        fit(F, feats, [0, 1, 2, 3, 4]).save_model(os.path.join(RESULTS, "final", "final_v2", f"stage1_{tag}_all_folds.json"))
    os.makedirs(os.path.join(RESULTS, "xgboost", "xgb_c2_blend_v2"), exist_ok=True)
    json.dump({"blend": {"full": "xgb_c2_sim19_v2", "noemb": "xgb_c2_sim19_noemb_v2", "w_seen": 0.75, "w_unseen": 0.5}},
              open(os.path.join(RESULTS, "xgboost", "xgb_c2_blend_v2", "report.json"), "w"))
    o1, o2 = oof(F, *mods["xgb_c2_sim19_v2"]), oof(F, *mods["xgb_c2_sim19_noemb_v2"])
    OB = o1.with_columns((0.75 * pl.col("p") + 0.25 * o2["p"]).alias("p"))
    OB.write_parquet(os.path.join(CACHE, "feats", "oof_xgb_c2_blend_v2.parquet"))
    # XLM-R scores on the band
    band = OB.filter((pl.col("p") >= 0.02) & (pl.col("p") <= 0.98))
    band.select(["a", "b", (pl.col("y") * 2.0 - 1 + pl.lit(rng.normal(0, 1, band.height))).alias("mlxenc")]).write_parquet(
        os.path.join(CACHE, "xenc", "ml_hardv2_train_scores.parquet"))
    # GNN (with / without XLM-R), saved where V3's edge_gnn.py puts its checkpoints
    for tag, extras in (("gnn_mlx_v2", [("xenc/ml_hardv2_train_scores.parquet", "mlxenc")]), ("gnn_v2", [])):
        G = GL.build_graph("c2_train_sim19", OB.select(["a", "b", "p"]), dev, extras)
        lg = GL.crossfit(G, os.path.join(CACHE, "gnn", tag), epochs=1, batch_s1=200)
        G.M.select(["a", "b", "fold", "y"]).with_columns(pl.Series("p", (1 / (1 + np.exp(-lg))).astype(np.float32))) \
            .write_parquet(os.path.join(CACHE, "feats", f"oof_{tag}.parquet"))
    # stage 2 (V2) on the blend
    n1 = len(s1)
    nr = len(rid)
    E = torch.nn.functional.normalize(torch.randn(n1 + nr, 16), dim=1).half()
    NAMEm = sp.random(n1 + nr, HN, density=0.02, format="csr", random_state=1, dtype=np.float32)
    ADDRm = sp.random(n1 + nr, HA, density=0.02, format="csr", random_state=2, dtype=np.float32)
    D = context_features(OB.select(["a", "b", "p"]).with_row_index("r"))
    D = sibling_features(D, E, NAMEm, ADDRm, n1, dev).sort("r")
    new = [c for c in D.columns if c not in ("r", "a", "b", "fold", "y")]
    S2 = pl.concat([F, D.select(new).rename({"p": "s1_p"})], how="horizontal")
    f2 = [c for c in noemb + [c if c != "p" else "s1_p" for c in new] if c not in ("sib_emb_max", "sib_emb_mean")]
    rd = os.path.join(RESULTS, "xgboost", "xgb_c2_blend_s2_v2")
    os.makedirs(rd, exist_ok=True)
    s2A, s2B = fit(S2, f2, [1, 2]), fit(S2, f2, [3, 4])
    s2A.save_model(os.path.join(rd, "model_A.json"))
    json.dump({"features": f2, "params": PARAMS, "rounds": 30}, open(os.path.join(rd, "report.json"), "w"))
    oof(S2, s2A, s2B, f2).write_parquet(os.path.join(CACHE, "feats", "oof_xgb_c2_blend_s2_v2.parquet"))
    dec = {"expected_f": {"best": {"gamma": 1.0, "extra": 0.0, "f05_tune": 0.9}}, "threshold_rule": {"thr": 0.7, "margin": 0.5, "f05_tune": 0.8}}
    for p in ("xgboost/xgb_c2_blend_s2_v2", "gnn/gnn_mlx_v2", "gnn/gnn_v2"):
        os.makedirs(os.path.join(RESULTS, p), exist_ok=True)
        json.dump(dec, open(os.path.join(RESULTS, p, "decode_eval.json"), "w"))

    # ---------------- test split
    s1t, c1t, ridt, rct, rec_t, Ct, yt = make_split("te", 1200, np.array(["US", "India", "France"]))
    save_sources("test", s1t, c1t, ridt, rct)
    n1t, nrt = len(s1t), len(ridt)
    NAMEt = sp.random(n1t + nrt, HN, density=0.02, format="csr", random_state=3, dtype=np.float32)
    ADDRt = sp.random(n1t + nrt, HA, density=0.02, format="csr", random_state=4, dtype=np.float32)
    sp.save_npz(os.path.join(CACHE, "blocking", "b1_test", "tfidf_name.npz"), NAMEt)
    sp.save_npz(os.path.join(CACHE, "blocking", "b1_test", "tfidf_addr.npz"), ADDRt)
    Ct.write_parquet(os.path.join(CACHE, "cands", "c2_test.parquet"))
    NM, AD = GCSR(NAMEt, dev), GCSR(ADDRt, dev)
    HAS = torch.from_numpy((np.diff(ADDRt.indptr) > 0).astype(np.float32))
    cs = []
    for f in range(5):
        torch.manual_seed(f)
        m = Encoder(HN, HA)
        torch.save(m.state_dict(), os.path.join(CACHE, "emb", "models", f"e3_fold{f}.pt"))
        Em = encode_rows(m, NM, AD, HAS, torch.arange(n1t + nrt))
        cs.append((Em[torch.from_numpy(Ct["a"].to_numpy().astype(np.int64))].float() *
                   Em[torch.from_numpy(Ct["b"].to_numpy().astype(np.int64) + n1t)].float()).sum(1).numpy())
    mean5 = np.mean(cs, 0)
    Ct.with_columns(pl.Series("cos_e3", mean5.astype(np.float32))).write_parquet(os.path.join(CACHE, "emb", "e3_test", "paircos_c2.parquet"))
    Ft = features(Ct, yt, mean5).with_columns(pl.Series("fold", np.array([fold_of(x) for x in s1t], np.int8)[Ct["a"].to_numpy()]))
    Ft = Ft.rename({"r_e3_emb": "r_e2f_emb"})
    write_parts(Ft, os.path.join(CACHE, "feats", "c2_test"))
    np.save(os.path.join(CACHE, "emb", "e2f_test", "emb.npy"), torch.nn.functional.normalize(torch.randn(n1t + nrt, 16), dim=1).half().numpy())
    Fr = Ft.rename({"r_e2f_emb": "r_e3_emb"})
    w = np.where(c1t[Fr["a"].to_numpy()] == "France", 0.5, 0.75)
    for kind, name in (("refit", "test_stage1_final_v2"), ("A", "test_stage1_halfA_v2")):
        ps = []
        for tag in ("xgb_c2_sim19_v2", "xgb_c2_sim19_noemb_v2"):
            mA, mB, feats = mods[tag]
            m = xgb.Booster(model_file=os.path.join(RESULTS, "final", "final_v2", f"stage1_{tag}_all_folds.json")) if kind == "refit" else mA
            ps.append(pred(m, Fr, feats))
        Fr.select(["a", "b"]).with_columns(pl.Series("p", (w * ps[0] + (1 - w) * ps[1]).astype(np.float32))) \
            .write_parquet(os.path.join(CACHE, "feats", f"{name}.parquet"))
    T1 = pl.read_parquet(os.path.join(CACHE, "feats", "test_stage1_final_v2.parquet"))
    tb = T1.filter((pl.col("p") >= 0.02) & (pl.col("p") <= 0.98))
    tb.select(["a", "b", pl.lit(rng.normal(0, 1.5, tb.height)).cast(pl.Float32).alias("mlxenc")]).write_parquet(
        os.path.join(CACHE, "xenc", "ml_hardv2_test_scores.parquet"))
    for tag, extras in (("gnn_mlx_v2", [("xenc/ml_hardv2_test_scores.parquet", "mlxenc")]), ("gnn_v2", [])):
        ck = GL.load(os.path.join(CACHE, "gnn", f"{tag}_A.pt"), "cpu")
        G = GL.build_graph("c2_test", T1, dev, extras, mu_sd=(ck["mu"].numpy(), ck["sd"].numpy()), with_labels=False)
        lg = GL.score_with(G, [os.path.join(CACHE, "gnn", f"{tag}_{h}.pt") for h in "AB"], batch_s1=200)
        G.M.select(["a", "b"]).with_columns(pl.Series("gnn", lg)).write_parquet(os.path.join(CACHE, "gnn", f"{tag}_test_scores.parquet"))
    # V2 test scores (stage 2 on the refit stage 1)
    Dt = context_features(T1.with_row_index("r"))
    Et = torch.from_numpy(np.load(os.path.join(CACHE, "emb", "e2f_test", "emb.npy")))
    Dt = sibling_features(Dt, Et, NAMEt, ADDRt, n1t, dev).sort("r")
    S2t = pl.concat([Fr, Dt.select(new).rename({"p": "s1_p"})], how="horizontal")
    S2t.select(["a", "b"]).with_columns(pl.Series("p", pred(s2A, S2t, f2))).write_parquet(os.path.join(CACHE, "feats", "test_scores_final_v2.parquet"))
    print("synthetic cache ready:", CACHE, "| train pairs", F.height, "| test pairs", Ft.height, flush=True)


if __name__ == "__main__":
    main()
