# Prompt for the coding agent on the H100: full V6 run (reuses the existing cache)

Copy everything between the lines into the agent.

---

You are on the H100 machine where this entity-resolution pipeline has already been run: V2, V3 and phase 2
(V4). V5 was NOT run. The cache is in `data/cache_v2/`; reports and models are in `results/`. The user pasted the
updated code (`A_ML_code_v6.zip`: `code/`, `run_all.sh`, docs) over the repository. Do not delete or regenerate
`data/`, `results/` or `submissions/`.

## 0. Verify the code update
```bash
ok=1
for f in code/v6/collective.py code/v6/train_collective.py code/v6/predict_v6.py code/v6/final_v6.py code/v6/pl_v6.py \
         code/v6/choose_v6.py code/v6/decode_v6.py code/v6/prior_em.py code/v6/simulate_universe.py code/v6/score_chain.py \
         code/v6/tests/smoke_v6.sh V6.md; do [ -f "$f" ] || { echo "MISSING $f"; ok=0; }; done
grep -q "v6all" run_all.sh || { echo "run_all.sh is the OLD version"; ok=0; }
grep -q "absent_code" code/v5/extra_feats.py || { echo "code/v5/extra_feats.py is the OLD version"; ok=0; }
grep -q -- "--model" code/neural_reranker/ml_cross_encoder.py || { echo "ml_cross_encoder.py is the OLD version"; ok=0; }
[ $ok = 1 ] && echo "code is up to date"; find code -name __pycache__ -type d -prune -exec rm -rf {} +
```
If anything is missing, stop and tell the user. Optional CPU self-test (temporary folder, ~3 min):
`bash code/v6/tests/smoke_v6.sh /tmp/er_smoke_v6` must end with `V6 SMOKE TEST PASSED (35 runner steps)`.
Read `V6.md` sections 0, 4, 5 and 6.

## 1. Inputs that must already exist
- `data/cache_v2/feats/c2_train/` and `data/cache_v2/feats/c2_test/` (the V2 feature parts)
- `data/cache_v2/drop_19.npy`
- `data/cache_v2/emb/e2f_{train,test}/emb.npy`
- `data/cache_v2/blocking/b1_{train,test}/tfidf_{name,addr}.npz` and `rid_{name,addr}_{idx,sc}.npy`
- `data/cache_v2/norm_v2_{train,test}_s{1,2,3}.parquet`
- `results/xgboost/xgb_c2_sim19_v2/`, `xgb_c2_sim19_noemb_v2/`, `xgb_c2_blend_v2/`, `xgb_c2_blend_s2_v2/` (models + reports)
- `data/cache_v2/feats/oof_xgb_c2_blend_v2.parquet` and `oof_xgb_c2_blend_s2_v2.parquet`
- `data/cache_v2/xenc/ml_hardv2_train_scores.parquet` and `ml_hardv2_test_scores.parquet` (V3 XLM-R)
- `results/gnn/gnn_mlx_v2/decode_eval.json`
- `data/cache_v2/p2/decisions/V3_noGNN_test.parquet` (phase-2 ablation)
- `data/cache_v2/feats/test_stage1_final_v2.parquet` and `results/final/final_v2/test_stats.json`

If one is missing, report which one. Do not rebuild V2/V3 unless the user says so.
Also check that the machine can reach huggingface.co: V6 downloads `BAAI/bge-reranker-v2-m3` (2.3 GB). If it
can't, set `XENC2_MODEL=none` (V6 then uses the V3 XLM-R scores only).

You need about 150 GB of free disk (`df -h .`).

## 2. Run
```bash
nohup bash run_all.sh --stage v6all > run_v6.out 2>&1 &     # ~11-14 h; progress: tail -f run_v6.out
```
Rules:
* Environment fixes only: never change models, features, thresholds, the rules in `code/v6/choose_v6.py`, or
  anything in `submissions/`.
* Do not upload anything to the leaderboard. Put no large files in git.
* If a step fails, read `results/_runs/<step>.log`, fix only the environment (package, memory, disk), then run the
  same command again (finished steps are skipped).
* If `v6x_train` prints `WARNING: second cross-encoder failed`, that is allowed: V6 continues without it. Report it.
* Short on time? Skip the LOCO step with `touch data/cache_v2/_done/v6_loco` before it starts (the plan then uses no
  pseudo-labels and one gamma).
* After about 1.5 h, send the user `results/v6/score_chain_dmsA.json` (early answer: does the density universe
  reproduce the test gap, and what is the ceiling?).

## 3. Report back
1. `results/v6/v6_choice.json` in full: `first_choice_file`, `submit_order`, `dms_min_f0_f4`, `checks`.
2. `results/v6/v6_plan.json` (the `plan`, `em_evidence`, `gamma_evidence`, `pl_evidence` fields).
3. `results/v6/score_chain_dmsA.json` in full.
4. `results/xgboost/xgb_v6_c1/decode_eval.json` and `xgb_v6_c2/decode_eval.json` (`expected_f.val`, `expected_f.best`).
5. `results/neural_reranker/ml_v6band/report_train.json` (if the second cross-encoder ran).
6. `results/final/final_v6/test_stats.json` (and `final_v6pl`, `final_v3nognn` if present).
7. Validator results: `results/_runs/v6_val.log`.
8. Make a results zip without large files:
   `zip -r v6_results.zip results/v6 results/xgboost/xgb_v6_* results/neural_reranker/ml_v6band results/phase2/loco_India2US_v6* results/final/*/test_stats.json results/final/final_v6/predict_report.json results/_runs -x '*.parquet' '*.npy' '*.pt' '*.tsv' '*model_*.json' '*refit_*.json' '*pl_stage1_*.json'`

Keep every `results/final/*/output/` folder locally. Do not submit.

---
