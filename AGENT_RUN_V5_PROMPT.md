# Prompt for the coding agent on the H100: run V5 reusing everything already computed

Copy everything between the lines into the agent.

---

You are on the H100 machine where this entity-resolution pipeline has already been run (V2, V3, phase 2). The
cache from that run is in `data/cache_v2/` (plus `data/cache/`), and the reports and models are in `results/`. The
code has been updated to V5. Your job: run V5 on the full data, **reusing every existing artifact that V5 can
use**, train only what is new, report the results, and never break or overwrite the existing ones.

## 0. Update the code without touching data or results
- If this is a git checkout: `git fetch origin && git checkout claude/dazzling-gauss-3jgxpd && git pull`.
- If you were given `A_ML_code_v5.zip`: unzip it **over** the repository folder. It only contains `code/`,
  `run_all.sh`, `requirements.txt` and docs. Do not delete `data/`, `results/` or `submissions/`.
- Read `V5.md` (what V5 is and why) and `H100_MANUAL.md` (how the runner works), then run `bash run_all.sh --list`.

## 1. What is REUSED (must already exist) and what is NEW
Reused from the previous run: normalised tables, TF-IDF blocking lists, candidate pairs, bi-encoder embeddings,
the C2 feature tables, the SIM19 simulation, and the V2 models/OOF tables (V5 is compared against V2).
V5 adds new feature columns to *copies* of the feature tables (`feats/c2v5_*`) and trains new models
(`xgb_v5_*`). Old files are never overwritten. GNN / XLM-R / V3 / V4 artifacts are not used by V5.

Check that everything reusable is present (run from the repository folder):
```bash
C=data/cache_v2; miss=0
for f in $C/norm_v2_train_s1.parquet $C/norm_v2_train_s2.parquet $C/norm_v2_train_s3.parquet \
         $C/norm_v2_test_s1.parquet $C/norm_v2_test_s2.parquet $C/norm_v2_test_s3.parquet \
         $C/blocking/b1_train/tfidf_name.npz $C/blocking/b1_train/tfidf_addr.npz \
         $C/blocking/b1_train/rid_name_idx.npy $C/blocking/b1_train/rid_name_sc.npy $C/blocking/b1_train/rid_addr_idx.npy $C/blocking/b1_train/rid_addr_sc.npy \
         $C/blocking/b1_test/tfidf_name.npz $C/blocking/b1_test/tfidf_addr.npz \
         $C/blocking/b1_test/rid_name_idx.npy $C/blocking/b1_test/rid_name_sc.npy $C/blocking/b1_test/rid_addr_idx.npy $C/blocking/b1_test/rid_addr_sc.npy \
         $C/emb/e2f_train/emb.npy $C/emb/e2f_test/emb.npy $C/drop_19.npy \
         $C/feats/c2_train_sim19/part_000.parquet $C/feats/c2_test/part_000.parquet \
         $C/feats/oof_xgb_c2_blend_s2_v2.parquet data/cache/train_gt_pairs.parquet \
         results/xgboost/xgb_c2_sim19_v2/report.json results/xgboost/xgb_c2_sim19_noemb_v2/report.json \
         results/xgboost/xgb_c2_blend_s2_v2/decode_eval.json data/dataset/test/test_source1.tsv; do
  [ -e "$f" ] || { echo "MISSING $f"; miss=1; }; done; [ $miss = 0 ] && echo "all reusable inputs present"
```
- If `data/dataset/...` is missing, link the data: `bash run_all.sh --data <DATASET_FOLDER> --dry`.
- If anything else is missing, run `bash run_all.sh --data <DATASET_FOLDER> --stage v2`. It **skips** every V2 step
  whose output already exists and rebuilds only the missing pieces. Then re-check.
- Check that the feature tables are complete (a killed earlier run can leave them half-written): the number of
  `part_*.parquet` files must be **43** in `feats/c2_train_sim19/` (and in `feats/c2_train/`) and **42** in
  `feats/c2_test/` (`ls data/cache_v2/feats/c2_test | wc -l`). If a table is incomplete, rebuild only that step:
  `bash run_all.sh --only feats_train`, then `--only sim19` (train), or `--only feats_test` (test).

## 2. Run V5 (full data, new steps only)
```bash
nohup bash run_all.sh --stage v5all > run_v5.out 2>&1 &
tail -f run_v5.out
```
This runs, in order, only V5 steps (none of them have a done marker yet):
`v5` (new features → V5 stage 1 full + no-embedding → blend → stage 2 → decoder settings → error report → test scores),
`loco5` (India → US with V2 features and with V5 features, pseudo-label check, unseen-country γ),
`v5pl` (France pseudo-labels, stage-1 refit, France re-scored), and `v5final` (two submissions, validator, choice).

Optional, to save wall-clock time on a GPU with ≥ 80 GB free: after the `v5` block finishes, `loco5` and `v5pl` are
independent. You can run them at the same time in two terminals (`bash run_all.sh --stage loco5` and
`bash run_all.sh --stage v5pl`), then `bash run_all.sh --stage v5final`. Don't run two copies of the same block.

Expected time on a full H100: about 4–6 h (v5 ≈ 2 h, loco5 ≈ 2 h, v5pl ≈ 1 h, v5final ≈ 15 min).

## 3. If a step fails
- Read `results/_runs/<step>.log` and `H100_MANUAL.md` section 7. Re-running the same command resumes.
- Allowed: install packages, free disk (`data/cache_v2/p2/`, `data/cache_v2/emb/e3_test/paircos5_c2.parquet`, and
  the GNN/XLM-R caches of V3/V4 are not needed by V5), lower memory settings, re-run a single step with `--only`.
- A V5 step that was interrupted half way: delete its partial output folder (for example
  `data/cache_v2/feats/c2v5_test/` for `v5_feats_test`) and re-run it with `--only <step>`.
- NOT allowed: changing models, features, thresholds, folds, the selection rules (`code/v5/choose_v5.py`,
  `code/phase2/select_v4.py`), editing results by hand, deleting V2 artifacts, or anything in `submissions/`.
- If a fix would need a code change beyond the environment, stop and report the error with the last 50 log lines.

## 4. Hard rules
- Never overwrite or delete `submissions/`, the V2 outputs (`results/final/final_v2/`,
  `data/cache_v2/feats/*_v2*`, `results/xgboost/*_v2/`), or the dataset.
- Do not upload anything to the leaderboard.
- Do not commit large files (`data/`, `*.parquet`, `*.npy`, `*.pt`, `*.tsv` > 50 MB).

## 5. Report back
1. `results/v5/v5_choice.json` (all of it): which file to submit first and why.
2. `results/v5/recall_xgb_v5_s2.json`: the `f0|ALL`, `f0|US`, `f0|India` entries.
3. `results/phase2/loco_India2US_v2.json` and `loco_India2US_v5.json`: `component_deltas_iso_ef`, and for every
   system `iso+ef` → `US_all_unseen` / `India_f0_in_domain` F0.5 and `best_gamma_unseen`.
4. `results/phase2/loco_India2US_v5/pl_step.json`, `results/v5/pl_select.json`, `results/v5/pl_score.json`.
5. `results/final/final_v5/test_stats.json` and `results/final/final_v5pl/test_stats.json`, and the validator
   result (PASS or issues) from `results/_runs/val_v5.log`.
6. Zip the small files and hand them over:
   `zip -r v5_results.zip results/v5 results/phase2 results/final/final_v5*/test_stats.json results/_runs -x '*.parquet' '*.npy' '*.pt' '*.tsv'`
Keep `results/final/final_v5*/output/matching_results.tsv` locally; do not submit them.

---
