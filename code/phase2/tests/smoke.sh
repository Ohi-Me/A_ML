#!/usr/bin/env bash
# CPU smoke test of every phase-2 script on a synthetic cache (numbers are meaningless; this checks the plumbing).
#   bash code/phase2/tests/smoke.sh [WORKDIR]
# The XLM-R scorer needs the Hugging Face model, so its output is mocked with the same schema.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke}"
rm -rf "$SR"; mkdir -p "$SR"
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
cd "$REPO"
run(){ echo "=== $*"; python3 "$@" 2>&1 | grep -v -i warn | tail -n 4; }
XTR="xenc/ml_hardv2_train_scores.parquet:mlxenc"; XTE="xenc/ml_hardv2_test_scores.parquet:mlxenc"
run code/phase2/tests/make_synth.py
python3 - <<'PY'
import os, numpy as np, polars as pl
from common.decide import Scorer
C = os.environ["ER_CACHE"]
sc = Scorer("train", np.load(f"{C}/drop_19.npy"))
D = pl.read_parquet(f"{C}/feats/oof_xgb_c2_blend_v2.parquet")
f = sc.metrics(D.filter(pl.col("y") == 1).select(["a", "b"]), sc.fold == 0)[0]["macro_f05"]
print("perfect-prediction F0.5 on fold 0 (must be ~1, candidate recall < 1):", f); assert f > 0.9
PY
for s in "cos" "variants --vars mean5,hash1" "report" "stage1 --var hash1 --models refit,halfA" "stage1 --var mean5 --models refit"; do
  run code/phase2/cos_variants.py $s; done
run code/phase2/gnn_run.py train --tag gnn_mlx_nocos --extra $XTR --drop_feats cos_e3 --epochs 1 --batch_s1 200
run code/phase2/gnn_run.py score --ckpt p2/gnn/gnn_mlx_nocos --split test --s1 test_stage1_final_v2.parquet --extra $XTE --out gnn_mlx_nocos --batch_s1 200
run code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --split test --s1 test_stage1_final_v2.parquet --extra $XTE --override hash1 --out v3_gnnhash1 --batch_s1 200
run code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --halves B --split train --score_folds 0 --extra $XTR --out chainB_f0 --batch_s1 200
run code/phase2/v4.py prep
python3 - <<'PY'
import os, numpy as np, polars as pl          # mock of neural_reranker/xenc_score_pairs.py (same output schema)
C = os.environ["ER_CACHE"]
for src, jobs, out in (("p2/v4_pairs_halfA_hash1.parquet", {"xA": "pA"}, "p2/xenc_chainA_hash1.parquet"),
                       ("p2/v4_pairs_refit_hash1.parquet", {"xA": "pR", "xB": "pR"}, "p2/xenc_refit_hash1.parquet")):
    D = pl.read_parquet(os.path.join(C, src)); rng = np.random.default_rng(0)
    cols = [pl.when((pl.col(c) >= 0.02) & (pl.col(c) <= 0.98)).then(pl.lit(rng.normal(0, 1, D.height))).otherwise(None)
            .cast(pl.Float32).alias(o) for o, c in jobs.items()]
    D.with_columns(cols).filter(pl.any_horizontal([pl.col(o).is_not_null() for o in jobs])).select(["a", "b"] + list(jobs)) \
     .write_parquet(os.path.join(C, out))
PY
run code/phase2/v4.py xmean
run code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --split test --s1 p2_test_s1_refit_hash1.parquet --extra p2/xenc_refit_hash1_mlx.parquet:mlxenc --override hash1 --out v3_chainhash1 --batch_s1 200
run code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --halves A --split test --s1 p2_test_s1_halfA_hash1.parquet --extra p2/xenc_chainA_hash1_mlx.parquet:mlxenc --override hash1 --out v4a_chainA_hash1 --batch_s1 200
run code/phase2/gnn_run.py score --ckpt gnn/gnn_v2 --halves A --split test --s1 p2_test_s1_halfA_hash1.parquet --override hash1 --out v4b_gnnA_hash1 --batch_s1 200
run code/phase2/v4.py stage2 --var hash1
run code/phase2/ablation.py --n_boot 20
run code/phase2/diff_accepts.py
for extra in "--noemb" ""; do
  run code/phase2/loco_eval.py stage1 --rounds 30 $extra
  python3 - $extra <<'PY'
import sys, numpy as np                        # XLM-R mocked: same data flow, trivial model
import neural_reranker.ml_cross_encoder as M
M.tokenize = lambda t1, t2, bs=0: (np.array([[len(a) % 7, len(b) % 5] for a, b in zip(t1, t2)], np.int32), 0)
M.train = lambda ids, pad, y, dev, epochs, seed=0: None
M.score = lambda m, ids, pad, dev, bs=0: np.random.default_rng(0).normal(0, 1, len(ids)).astype(np.float32)
sys.argv = ["loco_eval.py", "xenc"] + sys.argv[1:]
import phase2.loco_eval as L; L.main()
PY
  run code/phase2/loco_eval.py gnn --epochs 1 --batch_s1 200 $extra
  run code/phase2/loco_eval.py evaluate $extra
done
run code/phase2/select_v4.py
echo "SMOKE TEST PASSED"
