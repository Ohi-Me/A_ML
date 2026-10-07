# Prompt for the coding agent on the H100: V8 run (on the existing V6 cache)

Copy everything between the lines into the agent.

---

You are on the H100 machine where this entity-resolution pipeline has already run V2, V3, phase 2 and V6 (and maybe
V7). The cache is in `data/cache_v2/`; reports and models are in `results/`. The user updated the code (git pull of
branch `v8-work`, or the zip unpacked over the repository). Do not delete `data/`, `results/` or `submissions/`.

## 0. Verify
```bash
ok=1
for f in code/v8/prune.py code/v8/sib_probe.py code/v8/choose_v8.py code/v8/tests/smoke_v8.sh V8.md; do
  [ -f "$f" ] || { echo "MISSING $f"; ok=0; }; done
grep -q "v8all" run_all.sh || { echo "run_all.sh is the OLD version"; ok=0; }
grep -q "c_name_same" code/v6/collective.py || { echo "collective.py is OLD"; ok=0; }
[ $ok = 1 ] && echo "code is up to date"; find code -name __pycache__ -type d -prune -exec rm -rf {} +
ls results/v6/v6_choice.json data/cache_v2/xenc/ml_hardv2_model_A.pt data/cache_v2/emb/e3_train
df -h . ; free -g
```
Optional CPU self-test (~15 min): `bash code/v8/tests/smoke_v8.sh /tmp/er_smoke_v8` must end with
`V8 SMOKE TEST PASSED`. Read `V8.md`.

If V7 is still running, let it finish first (V8 reuses nothing V7 writes, but both need the GPU and disk).
Free disk needed: ~150 GB. RAM: the V8 steps are built to fit a 32 GB job; if one runs out of memory, lower
`V8_BUDGET` (e.g. `V8_BUDGET=18`), delete `data/cache_v2/_done/v8_cands_*` and `data/cache_v2/cands/c3w*`, and re-run.

## 1. Run
```bash
nohup bash run_all.sh --stage v8all > run_v8.out 2>&1 &
```
* **As soon as `v8_miss_probe` is done, send the user:** `results/v8/sib_probe.json`, `results/v7/miss_probe.json`,
  `results/v7/missed_sample.tsv`.
* **If `sib_probe.json → id_check.spearman_true_pairs` is clearly different from `spearman_random_pairs`**
  (|difference| > 0.05), stop the run and tell the user. Nothing in the pipeline reads ids.
* **After `v8_prune_test`, send:** `results/v8/prune_c3wdms.json`, `results/v8/prune_apply_c4dms_train.json`,
  `results/v8/prune_apply_c4_test.json`, `results/v7/cands_c3wdms_train.json`, `results/v7/cands_c3w_test.json`.
* Environment fixes only; never change features, models, rules or `submissions/`; no leaderboard uploads.
* If a step fails, fix the environment and run the same command again (finished steps are skipped).
* Optional stronger reranker (+2–3 h), only if the user asks: `XENC3_MODEL=Qwen/Qwen3-Reranker-0.6B bash run_all.sh --stage v8all`.

## 2. Report back
1. `results/v8/v8_choice.json` in full, and `results/v8/v8_plan.json` (`plan`, `ceiling_v8`, `candidates`, `pruner`).
2. `results/xgboost/xgb_v8_c1/decode_eval.json` and `xgb_v8_c2/decode_eval.json`.
3. `results/final/final_v8/test_stats.json` and `results/_runs/v8_val.log`.
4. A zip without large files:
   `zip -r v8_results.zip results/v8 results/v7/cands_c3w* results/v7/miss_probe.json results/xgboost/xgb_v8_* results/neural_reranker/ml_v8* results/final/final_v8/test_stats.json results/_runs -x '*.parquet' '*.npy' '*.pt' '*model_*.json' '*refit_*.json' '*pruner_*.json'`

Keep `results/final/final_v8/output/` locally. Do not submit.

---
