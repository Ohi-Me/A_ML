#!/usr/bin/env bash
# CPU smoke test of the whole V6 chain on a synthetic cache (numbers are meaningless; this checks the plumbing).
# It runs the EXACT step commands of run_all.sh (blocks v6diag, v6a, v6feat ... v6pick, taken from --list), with two
# stand-ins: the cross-encoder (no Hugging Face download) and the official validator (needs the challenge TSVs).
# The France pseudo-label steps are also forced once, because the plan normally switches them off on synthetic data.
#   bash code/v6/tests/smoke_v6.sh [WORKDIR]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SR="${1:-/tmp/er_smoke_v6}"
rm -rf "$SR"; mkdir -p "$SR/results/_runs"
export ER_ROOT="$SR" ER_CACHE="$SR/data/cache_v2" ER_NORM=v2 ER_ALLOW_CPU=1 PYTHONPATH="$REPO/code" PYTHONWARNINGS=ignore
cd "$REPO"
run(){ echo "=== $1"; bash -c "$2" > "$SR/results/_runs/$1.log" 2>&1 || { echo "FAILED: $2"; tail -n 40 "$SR/results/_runs/$1.log"; exit 1; }
       grep -v -i warn "$SR/results/_runs/$1.log" | tail -n 2; }
run make_synth "python3 code/phase2/tests/make_synth.py"
python3 - <<'PY'
import os, polars as pl
C = os.environ["ER_CACHE"]; os.makedirs(f"{C}/p2/decisions", exist_ok=True)
pl.read_parquet(f"{C}/feats/test_stage1_final_v2.parquet").filter(pl.col("p") > 0.5).select(["a", "b"]).write_parquet(f"{C}/p2/decisions/V3_noGNN_test.parquet")
PY
bash run_all.sh --list > "$SR/steps.txt"
# the runner's commands use paths relative to the repository root (code/..., results/...): run them from the work
# folder with code/ linked into it, so every output stays inside the work folder
ln -sfn "$REPO/code" "$SR/code"
cd "$SR"
# final_v2 test stats (the V2 reference of the consistency check) from the synthetic V2 test scores
run final_v2 "python3 code/v6/final_v6.py --scores feats/test_scores_final_v2.parquet --tag xgb_c2_blend_s2_v2 --out final_v2"
n=0
while read -r block name cmd; do
  [[ "$block" =~ ^(v6diag|v6a|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ ]] || continue
  cmd="${cmd//python -u code\/neural_reranker\/ml_cross_encoder.py/python -u code/v6/tests/mock_xenc.py}"
  cmd="${cmd//bash code\/final\/validate.sh/python -u code/v6/tests/mini_validate.py}"
  if [[ "$name" == v6_choose ]]; then        # force the pseudo-label path once before the final pick
    for s in select fit score; do run "force_pl_$s" "python -u code/v6/pl_v6.py $s"; done
    run force_pl_final "python -u code/v6/final_v6.py --scores feats/test_scores_final_v6pl.parquet --tag xgb_v6_c2 --caps --em --out final_v6pl"
    run force_pl_val "python -u code/v6/tests/mini_validate.py $SR/results/final/final_v6pl/output"
  fi
  run "$name" "$cmd"; n=$((n + 1))
done < "$SR/steps.txt"
[[ $n -ge 30 ]] || { echo "only $n V6 steps found in run_all.sh --list"; exit 1; }
python3 - <<'PY'
import json, os
R = os.path.join(os.environ["ER_ROOT"], "results")
c = json.load(open(f"{R}/v6/v6_choice.json")); p = json.load(open(f"{R}/v6/v6_plan.json"))
ch = json.load(open(f"{R}/v6/score_chain_dmsA.json"))
for k in ("sim19_as_validated", "universe_sim19_calibration", "universe_sim19_calibration_em", "universe_own_calibration",
          "v3nognn_universe_sim19_calibration", "ceiling", "ceiling_sim19"):
    assert k in ch, k
assert ch["ceiling"]["f0"] > 0.9, ch["ceiling"]
rep = json.load(open(f"{R}/xgboost/xgb_v6_c2/report.json"))
assert "bgexenc" in rep["features"] and "mlxenc" in rep["features"] and "prev_p" in rep["features"] and "c_support" in rep["features"]
print("plan", p["plan"]); print("DMS min(f0,f4)", c["dms_min_f0_f4"]); print("submit order", c["submit_order"])
for f in ("final_v6", "final_v6pl", "final_v6a", "final_v3nognn"):
    assert os.path.exists(f"{R}/final/{f}/output/matching_results.tsv"), f
PY
echo "V6 SMOKE TEST PASSED ($n runner steps)"
