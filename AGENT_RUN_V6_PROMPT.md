# Prompt for the coding agent on the H100: V6 diagnostics + V6a (reuses the existing cache)

Copy everything between the lines into the agent.

---

You are on the H100 machine where this entity-resolution pipeline has already been run (V2, V3, phase 2, possibly V5).
The cache is in `data/cache_v2/`, reports and models in `results/`. The user pasted the updated code
(`A_ML_code_v6.zip`: `code/`, `run_all.sh`, docs) over the repository. Do not delete or regenerate `data/`,
`results/` or `submissions/`.

## 0. Verify the code update
```bash
ok=1
for f in code/v6/data_probe.py code/v6/simulate_universe.py code/v6/score_chain.py code/v6/submit_from_decisions.py \
         code/v6/tests/smoke_v6.sh V6_RESEARCH.md; do [ -f "$f" ] || { echo "MISSING $f"; ok=0; }; done
grep -q "v6diag" run_all.sh || { echo "run_all.sh is the OLD version"; ok=0; }
[ $ok = 1 ] && echo "code is up to date"; find code -name __pycache__ -type d -prune -exec rm -rf {} +
```
If anything is missing, stop and tell the user. Optional CPU self-test (temporary folder):
`bash code/v6/tests/smoke_v6.sh /tmp/er_smoke_v6` must print `V6 SMOKE TEST PASSED`.
Read `V6_RESEARCH.md` sections 0, 2 and 7.

## 1. Inputs that must already exist
`data/cache_v2/feats/c2_train/` (43 parts), `data/cache_v2/drop_19.npy`, `data/cache_v2/emb/e2f_train/emb.npy`,
`data/cache_v2/blocking/b1_train/tfidf_{name,addr}.npz`, `results/xgboost/xgb_c2_sim19_v2/model_A.json` (+ `_B`),
`results/xgboost/xgb_c2_sim19_noemb_v2/model_{A,B}.json`, `results/xgboost/xgb_c2_blend_s2_v2/model_{A,B}.json`,
`data/cache_v2/feats/oof_xgb_c2_blend_s2_v2.parquet`, `data/cache_v2/p2/decisions/V3_noGNN_test.parquet`
(written by the phase-2 ablation), `data/cache_v2/feats/test_stage1_final_v2.parquet`.
If one is missing, report which one. Do not rebuild V2/V3 unless the user says so.

## 2. Run
```bash
nohup bash run_all.sh --stage v6diag > run_v6diag.out 2>&1 &   # ~1.5-2 h: probe, V3-without-GNN file, density universe, V2 chain on it
# when it has finished:
nohup bash run_all.sh --stage v6a > run_v6a.out 2>&1 &         # ~2-3 h: V2 recipe trained + calibrated at test density
```
Rules: same as before. Only environment fixes; never change models, features, thresholds, selection rules or
anything in `submissions/`; do not upload anything to the leaderboard; no large files in git.

## 3. Report back
1. `results/v6/data_probe.json` (all of it).
2. `results/v6/universe_c2_train_dmsA.json` and `results/v6/score_chain_dmsA.json` (all of it): F0.5 and
   pred_over_true per country for `sim19_as_validated`, `universe_sim19_calibration`, `universe_own_calibration`.
3. `results/xgboost/xgb_v6a_s2/decode_eval.json` (`expected_f.val`, `expected_f.best`) and
   `results/final/final_v6a/test_stats.json`, `results/final/final_v3nognn/test_stats.json`.
4. Validator results for `final_v3nognn` and `final_v6a` (`results/_runs/v6_v3nognn.log`, `v6a_val.log`).
5. `zip -r v6_results.zip results/v6 results/xgboost/xgb_v6a_* results/final/final_v6a/test_stats.json results/final/final_v3nognn/test_stats.json results/_runs -x '*.parquet' '*.npy' '*.pt' '*.tsv' '*model_*.json'`
Keep `results/final/final_v3nognn/output/` and `results/final/final_v6a/output/` locally; do not submit.

---
