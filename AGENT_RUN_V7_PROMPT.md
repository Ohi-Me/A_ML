# Prompt for the coding agent on the H100: V7 run (on the existing V6 cache)

Copy everything between the lines into the agent.

---

You are on the H100 machine where this entity-resolution pipeline has already run V2, V3, phase 2 and V6. The cache is
in `data/cache_v2/`; reports and models are in `results/`. The user updated the code (git pull of branch
`claude/dazzling-gauss-3jgxpd`, or the zip unpacked over the repository). Do not delete `data/`, `results/` or
`submissions/`.

## 0. Verify
```bash
ok=1
for f in code/v7/miss_probe.py code/v7/error_dump.py code/v7/build_cands.py code/v7/cands_plan.py code/v7/choose_v7.py \
         code/v7/keys.py code/v7/tests/smoke_v7.sh V7.md; do [ -f "$f" ] || { echo "MISSING $f"; ok=0; }; done
grep -q "v7all" run_all.sh || { echo "run_all.sh is the OLD version"; ok=0; }
grep -q "models_from" code/neural_reranker/ml_cross_encoder.py || { echo "ml_cross_encoder.py is OLD"; ok=0; }
[ $ok = 1 ] && echo "code is up to date"; find code -name __pycache__ -type d -prune -exec rm -rf {} +
ls results/v6/v6_choice.json data/cache_v2/feats/oof_xgb_v6_c2.parquet data/cache_v2/xenc/ml_hardv2_model_A.pt
df -h . ; free -g
```
Optional CPU self-test (~6 min): `bash code/v7/tests/smoke_v7.sh /tmp/er_smoke_v7` must end with
`V7 SMOKE TEST PASSED (33 runner steps)`. Read `V7.md`.

Prerequisites: ~200 GB of free disk and at least 128 GB RAM. If RAM is below 128 GB, run
`python code/v7/cands_plan.py --budget 10` once after `v7diag` and before `v7_cands_train`.

## 1. Run
```bash
nohup bash run_all.sh --stage v7all > run_v7.out 2>&1 &
```
* **As soon as `v7_error_dump` is done (~1 h), send the user these files:**
  * `results/v7/miss_probe.json`, `missed_sample.tsv`
  * `results/v7/error_dump.json`, `fn_sample.tsv`, `fp_sample.tsv`, `france_sample.tsv`
  * `results/v7/cands_plan.json`
* Environment fixes only; never change features, models, rules or `submissions/`; no leaderboard uploads.
* If a step fails, fix the environment and run the same command again (finished steps are skipped).
* If `pair_features` (v7_feats_*) runs out of memory:
  1. Run `python code/v7/cands_plan.py --budget 10`.
  2. Delete `data/cache_v2/_done/v7_cands_*` and `data/cache_v2/cands/c3*`.
  3. Re-run.
* Optional stronger reranker (+2–3 h): start with `XENC3_MODEL=Qwen/Qwen3-Reranker-0.6B bash run_all.sh --stage v7all`
  before the `v7xenc` block runs. Only if the user asks.

## 2. Report back
1. `results/v7/v7_choice.json` in full, and `results/v7/v7_plan.json` (`plan`, `ceiling_v7`, `dms`).
2. `results/v7/cands_c3dms_train.json` and `cands_c3_test.json` (recall, pairs per record, passes kept).
3. `results/xgboost/xgb_v7_c1/decode_eval.json` and `xgb_v7_c2/decode_eval.json`.
4. `results/neural_reranker/ml_v7xlmr/report_train.json` and `ml_v7bge/report_train.json`.
5. `results/final/final_v7/test_stats.json` and `results/_runs/v7_val.log`.
6. A zip without large files:
   `zip -r v7_results.zip results/v7 results/xgboost/xgb_v7_* results/neural_reranker/ml_v7* results/final/final_v7/test_stats.json results/_runs -x '*.parquet' '*.npy' '*.pt' '*model_*.json' '*refit_*.json'`

Keep `results/final/final_v7/output/` locally. Do not submit.

---
