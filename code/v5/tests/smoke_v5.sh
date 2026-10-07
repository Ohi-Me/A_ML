#!/usr/bin/env bash
# CPU smoke test of the whole V5 block on a synthetic cache (numbers are meaningless; this checks the plumbing,
# including the unchanged V2 trainers train_xgb.py / stage2.py / decode_eval.py / predict_test_v2.py on the V5 table).
#   bash code/v5/tests/smoke_v5.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke_v5}"
rm -rf "$SR"; mkdir -p "$SR"
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
cd "$REPO"
run(){ echo "=== $*"; python3 "$@" > "$SR/last.log" 2>&1 || { tail -n 30 "$SR/last.log"; exit 1; }; grep -v -i warn "$SR/last.log" | tail -n 3; }
DROP="cos_e3,b_gap_cos_e3,b_rank_cos_e3,a_gap_cos_e3,a_rank_cos_e3,b_gap2_cos_e3,r_e3_emb"
run code/phase2/tests/make_synth.py
run code/v5/extra_feats.py --src c2_train_sim19
run code/v5/extra_feats.py --src c2_test
run code/xgboost/train_xgb.py --feats c2v5_train_sim19 --tag xgb_v5_full --simdrop 19 --rounds 60
run code/xgboost/train_xgb.py --feats c2v5_train_sim19 --tag xgb_v5_noemb --simdrop 19 --rounds 60 --drop $DROP
run code/xgboost/blend_oof.py --full xgb_v5_full --noemb xgb_v5_noemb --w 0.75 --out xgb_v5_blend
run code/xgboost/stage2.py --feats c2v5_train_sim19 --s1tag xgb_v5_blend --tag xgb_v5_s2 --etag e2f --simdrop 19 --rounds 60 --drop_feats $DROP,sib_emb_max,sib_emb_mean
run code/xgboost/decode_eval.py --tag xgb_v5_s2 --simdrop 19
run code/v5/recall_report.py --s1 xgb_v5_blend --s2 xgb_v5_s2
run code/final/predict_test_v2.py --s1_feats c2v5_train_sim19 --s1_tag xgb_v5_full --blend_from xgb_v5_blend --s2_tag xgb_v5_s2 --test_feats c2v5_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v5
run code/phase2/loco_eval.py all --skip_neural --name v2 --rounds 60 --max_pl_pairs 100000
run code/phase2/loco_eval.py all --skip_neural --name v5 --feats_table c2v5_train_sim19 --full_tag xgb_v5_full --noemb_tag xgb_v5_noemb --rounds 60
run code/v5/pseudo_label.py select
run code/v5/pseudo_label.py fit
run code/v5/pseudo_label.py score
run code/v5/final_v5.py --scores feats/test_scores_final_v5.parquet --loco_system stage1_all --out final_v5
run code/v5/final_v5.py --scores feats/test_scores_final_v5pl.parquet --loco_system stage1_pl --out final_v5pl
run code/v5/choose_v5.py
echo "V5 SMOKE TEST PASSED"
