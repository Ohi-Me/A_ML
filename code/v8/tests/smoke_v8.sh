#!/usr/bin/env bash
# CPU smoke test of the whole V8 chain on a synthetic cache (numbers are meaningless; this checks the plumbing).
# Runs the V6 smoke test first (V8 builds on the V6 cache), then the EXACT step commands of run_all.sh for the V8 blocks
# (v8diag ... v8pick, from --list). Stand-ins: the cross-encoder model (no Hugging Face download; the V3 XLM-R
# checkpoints are copies of the V6 stand-in ones) and the official validator. Everything else - probes, wide candidate
# builder, pruner, pair features, universe, XGBoost, cross-encoder re-scoring plumbing, collective rounds, test chain,
# decoding, choice - is the real code.
#   bash code/v8/tests/smoke_v8.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke_v8}"
bash "$REPO/code/v6/tests/smoke_v6.sh" "$SR" | tail -n 1
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
run(){ echo "=== $1"; bash -c "$2" > "$SR/results/_runs/$1.log" 2>&1 || { echo "FAILED: $2"; tail -n 40 "$SR/results/_runs/$1.log"; exit 1; }
       grep -v -i warn "$SR/results/_runs/$1.log" | tail -n 2; }
for h in A B; do cp "$ER_CACHE/xenc/ml_v6band_model_$h.pt" "$ER_CACHE/xenc/ml_hardv2_model_$h.pt"; done
cd "$REPO" && bash run_all.sh --list > "$SR/steps8.txt"
cd "$SR"
n=0
while read -r block name cmd; do
  [[ "$block" =~ ^v8(diag|cands|prune|feat|s1|xenc|col|test|final|pick)$ ]] || continue
  cmd="${cmd//python -u code\/neural_reranker\/ml_cross_encoder.py/python -u code/v6/tests/mock_xenc.py}"
  cmd="${cmd//bash code\/final\/validate.sh/python -u code/v6/tests/mini_validate.py}"
  run "$name" "$cmd"; n=$((n + 1))
done < "$SR/steps8.txt"
[[ $n -ge 35 ]] || { echo "only $n V8 steps found"; exit 1; }
python3 - <<'PY'
import json, os
R = os.path.join(os.environ["ER_ROOT"], "results")
sp = json.load(open(f"{R}/v8/sib_probe.json"))
print("sib probe (normalised name, empty address):", sp["normalised_name"]["empty_address"], "id check:", sp["id_check"])
w = json.load(open(f"{R}/v7/cands_c3wdms_train.json"))
print("wide train candidates:", {k: w[k] for k in ("pairs", "pairs_per_record", "recall", "passes_kept")})
pr = json.load(open(f"{R}/v8/prune_c3wdms.json"))
print("pruner:", {k: pr[k] for k in ("tau", "pairs_per_record_before", "pairs_per_record_after")}, pr["by_fold"]["0"])
te = json.load(open(f"{R}/v8/prune_apply_c4_test.json"))
print("pruned test candidates:", {k: te[k] for k in ("pairs_before", "pairs_after", "pairs_per_record_after")})
assert pr["pairs_per_record_after"] <= pr["pairs_per_record_before"]
rep = json.load(open(f"{R}/xgboost/xgb_v8_c2/report.json"))
f = rep["features"]
assert "mlxenc" in f and "bgexenc" in f and "c_name_same" in f and any(x.startswith("x_") for x in f), f
c = json.load(open(f"{R}/v8/v8_choice.json"))
print("plan", c["plan"]); print("DMS", c["dms_min_f0_f4"], "ceiling", c["ceiling"]); print("order", c["submit_order"])
assert os.path.exists(f"{R}/final/final_v8/output/matching_results.tsv")
PY
echo "V8 SMOKE TEST PASSED ($n runner steps)"
