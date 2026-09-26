#!/usr/bin/env bash
# CPU smoke test of the whole V7 chain on a synthetic cache (numbers are meaningless; this checks the plumbing).
# Runs the V6 smoke test first (V7 builds on V6 outputs), then the EXACT step commands of run_all.sh for the V7 blocks
# (v7diag ... v7pick, from --list). Stand-ins: the cross-encoder model (no Hugging Face download; the V3 XLM-R
# checkpoints are copies of the V6 stand-in ones) and the official validator. Everything else - candidate builder,
# bi-encoder pair cosines, pair_features.py, universe, extra features, XGBoost, collective rounds, test chain,
# decoding, choice - is the real code.
#   bash code/v7/tests/smoke_v7.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke_v7}"
bash "$REPO/code/v6/tests/smoke_v6.sh" "$SR" | tail -n 1
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
run(){ echo "=== $1"; bash -c "$2" > "$SR/results/_runs/$1.log" 2>&1 || { echo "FAILED: $2"; tail -n 40 "$SR/results/_runs/$1.log"; exit 1; }
       grep -v -i warn "$SR/results/_runs/$1.log" | tail -n 2; }
for h in A B; do cp "$ER_CACHE/xenc/ml_v6band_model_$h.pt" "$ER_CACHE/xenc/ml_hardv2_model_$h.pt"; done
cd "$REPO" && bash run_all.sh --list > "$SR/steps7.txt"
cd "$SR"
n=0
while read -r block name cmd; do
  [[ "$block" =~ ^(v7diag|v7cands|v7feat|v7s1|v7xenc|v7col|v7test|v7final|v7pick)$ ]] || continue
  cmd="${cmd//python -u code\/neural_reranker\/ml_cross_encoder.py/python -u code/v6/tests/mock_xenc.py}"
  cmd="${cmd//bash code\/final\/validate.sh/python -u code/v6/tests/mini_validate.py}"
  run "$name" "$cmd"; n=$((n + 1))
done < "$SR/steps7.txt"
[[ $n -ge 30 ]] || { echo "only $n V7 steps found"; exit 1; }
python3 - <<'PY'
import json, os
R = os.path.join(os.environ["ER_ROOT"], "results")
cands = json.load(open(f"{R}/v7/cands_c3dms_train.json"))
print("V7 train candidates:", {k: cands[k] for k in ("pairs", "pairs_per_record", "recall", "recall_lists_only", "passes_kept")})
rep = json.load(open(f"{R}/xgboost/xgb_v7_c2/report.json"))
assert "mlxenc" in rep["features"] and "bgexenc" in rep["features"] and any(f.startswith("x_") for f in rep["features"]), rep["features"]
c = json.load(open(f"{R}/v7/v7_choice.json"))
print("plan", c["plan"]); print("DMS", c["dms_min_f0_f4"], "ceiling", c["ceiling"]); print("order", c["submit_order"])
assert os.path.exists(f"{R}/final/final_v7/output/matching_results.tsv")
PY
echo "V7 SMOKE TEST PASSED ($n runner steps)"
