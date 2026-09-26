#!/usr/bin/env bash
# CPU smoke test of the V6 diagnostics on a synthetic cache (numbers meaningless; checks the plumbing).
#   bash code/v6/tests/smoke_v6.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke_v6}"
rm -rf "$SR"; mkdir -p "$SR"
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
cd "$REPO"
run(){ echo "=== $*"; python3 "$@" > "$SR/last.log" 2>&1 || { tail -n 30 "$SR/last.log"; exit 1; }; grep -v -i warn "$SR/last.log" | tail -n 4; }
run code/phase2/tests/make_synth.py
run code/v6/data_probe.py
run code/v6/simulate_universe.py --src c2_train --plan US:0.62:0.19,India:1.0:0.19 --tag dmsA --code 62
run code/v6/score_chain.py --feats c2_train_dmsA --code 62 --name dmsA
python3 - <<'PY'
import os, polars as pl
C = os.environ["ER_CACHE"]; os.makedirs(f"{C}/p2/decisions", exist_ok=True)
pl.read_parquet(f"{C}/feats/test_stage1_final_v2.parquet").filter(pl.col("p") > 0.5).select(["a", "b"]).write_parquet(f"{C}/p2/decisions/V3_noGNN_test.parquet")
PY
run code/v6/submit_from_decisions.py --decisions p2/decisions/V3_noGNN_test.parquet --out final_v3nognn
DROP="cos_e3,b_gap_cos_e3,b_rank_cos_e3,a_gap_cos_e3,a_rank_cos_e3,b_gap2_cos_e3,r_e3_emb"
run code/xgboost/train_xgb.py --feats c2_train_dmsA --tag xgb_v6a_full --simdrop 62 --rounds 60
run code/xgboost/train_xgb.py --feats c2_train_dmsA --tag xgb_v6a_noemb --simdrop 62 --rounds 60 --drop $DROP
run code/xgboost/blend_oof.py --full xgb_v6a_full --noemb xgb_v6a_noemb --w 0.75 --out xgb_v6a_blend --simdrop 62
run code/xgboost/stage2.py --feats c2_train_dmsA --s1tag xgb_v6a_blend --tag xgb_v6a_s2 --etag e2f --simdrop 62 --rounds 60 --drop_feats $DROP,sib_emb_max,sib_emb_mean
run code/xgboost/decode_eval.py --tag xgb_v6a_s2 --simdrop 62
run code/final/predict_test_v2.py --s1_feats c2_train_dmsA --s1_tag xgb_v6a_full --blend_from xgb_v6a_blend --s2_tag xgb_v6a_s2 --test_feats c2_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v6a
echo "V6 SMOKE TEST PASSED"
