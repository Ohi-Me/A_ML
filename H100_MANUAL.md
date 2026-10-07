# H100 run manual — Business Entity Resolution (V2 → V3 → phase-2 diagnosis → V4 → V6)

**Current version: V7** (`V7.md`): on the V6 cache run `nohup bash run_all.sh --stage v7all > run_v7.out 2>&1 &`
(about 8–11 h; choice in `results/v7/v7_choice.json`; agent prompt `AGENT_RUN_V7_PROMPT.md`).

V6 (`V6.md`). On a machine where V2, V3 and phase 2 already ran:
`nohup bash run_all.sh --stage v6all > run_v6.out 2>&1 &` (about 11–14 h). The chosen file is named in
`results/v6/v6_choice.json` (`first_choice_file`). Agent prompt: `AGENT_RUN_V6_PROMPT.md`.

Everything runs on **one GPU machine** with **one command**. There are no hard-coded hosts, users, or paths, and you
don't need a scheduler. You can run it from any folder that contains this repository.

```bash
bash run_all.sh --data /path/to/dataset        # full pipeline, resumable
```

`/path/to/dataset` is the folder that holds the original challenge files:

```
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

The runner links it to `data/dataset`, which is where the code reads from. Nothing inside the dataset is modified.

---

## 1. Machine requirements

| | minimum | used before |
|---|---|---|
| GPU | NVIDIA with ≥ 40 GB, CUDA 12.x driver | H100 NVL MIG 3g.47gb slice (a full H100 80 GB is faster) |
| CPU / RAM | 8 cores / 64 GB (32 cores helps blocking + features) | 8–32 cores, 32 GB job cap |
| free disk | **250 GB** for `data/cache_v2` + `data/cache` | the cache reached ~100 GB for V2/V3 alone |
| internet | once, to download `FacebookAI/xlm-roberta-base` (MIT, 278 M) from Hugging Face | — |

## 2. One-time setup

```bash
git clone https://github.com/Ohi-Me/A_ML.git && cd A_ML          # or unzip the repo zip and cd into it
git checkout claude/dazzling-gauss-3jgxpd                         # the branch with phase 2 (skip for the zip)

conda create -y -n er python=3.10 && conda activate er            # or python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# optional, if the GPU node has no internet: download the cross-encoder once on a login node
HF_HOME=$PWD/hf_cache python -c "from transformers import AutoModel, AutoTokenizer as T; \
  m='FacebookAI/xlm-roberta-base'; AutoModel.from_pretrained(m); T.from_pretrained(m)"
```

Check the environment (prints the GPU and exits without running anything):

```bash
bash run_all.sh --data /path/to/dataset --dry
```

## 3. Running

```bash
bash run_all.sh --data /path/to/dataset            # everything: v2 → v3 → audit → phase2 → loco → select → v4 → V6
bash run_all.sh --stage v6all                      # V6 only, on an existing V2/V3/phase-2 cache (V6.md)
bash run_all.sh --stage phase2                     # a single block (after v2 and v3 exist)
bash run_all.sh --only ablation                    # a single step (re-runs it even if it was done)
bash run_all.sh --list                             # every step, its block and its exact command
nohup bash run_all.sh --data /path/to/dataset > run_all.out 2>&1 &     # detached; follow with: tail -f run_all.out
```

* **Resumable:** each step that succeeds writes `data/cache_v2/_done/<step>`. It's also skipped when its main
  output already exists. If the machine or a job gets killed, run the same command again.
* **Logs:** `results/_runs/<step>.log`. On a failure the runner prints the last 25 lines and stops.
* **Reusing the old cluster cache:** if your old working folder still exists (it had `data/cache_v2`,
  `data/cache` and `results/xgboost/*/model_*.json`), copy this repository's `code/`, `run_all.sh` and
  `requirements.txt` into that folder and run `bash run_all.sh` there. V2, V3 and the audit steps are then
  skipped and only phase 2 runs (about 6 h).
* **Environment variables** (all optional): `ER_CACHE` (default `data/cache_v2`), `NCPUS` (default all cores),
  `HF_HOME` (default `./hf_cache`).

### With a scheduler (optional)

The runner is a plain bash script, so any scheduler can wrap it. Replace `<...>` with your values.

```bash
# PBS
#PBS -N er_all
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
cd <repo folder> && source <conda.sh> && conda activate er && bash run_all.sh --data <dataset folder>

# SLURM
#SBATCH --job-name=er_all --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00
cd <repo folder> && source <conda.sh> && conda activate er && bash run_all.sh --data <dataset folder>
```

If the cluster kills long jobs, submit the same script again. It continues where it stopped.

## 4. What each block does and how long it takes

Times were measured on an H100 NVL MIG 3g slice; a full H100 is about 1.5–2× faster.

| block | steps | time | what it produces |
|---|---|---|---|
| `v2` | normalisation → TF-IDF blocking → n-gram bi-encoders (e3 cross-fitted, e2f) → C2 candidates → pair features → SIM19 test-prior simulation → XGB stage 1 (full + no-embedding, blend) → XGB stage 2 → isotonic + expected-F0.5 → test | ~4.6 h | `results/final/final_v2/output/` (validation F0.5 0.98696; leaderboard 0.98332) |
| `v3` | edge GNN, XLM-R cross-encoder on uncertain pairs, GNN + XLM-R, test scoring | ~1.6 h | `results/final/final_v3/output/` (validation 0.99078; leaderboard 0.982814) |
| `audit` | 25 Sep audit: test stage-1 scores of the half models, V3 GNN fed half-model stage 1 | ~40 min | `results/audit/*.json` |
| `phase2` | bi-encoder cosine experiment, ablation matrix, V4 candidates, stage-by-stage trace of the extra accepts | ~3.5 h | `results/phase2/cos/`, `ablation.json/.csv`, `diff_accepts.json` |
| `loco` | India → US leave-one-country-out, with and without embedding features | ~3 h | `results/phase2/loco_India2US*.json` |
| `select` | pre-registered V4 selection rule (see `code/phase2/select_v4.py`) | 1 min | `results/phase2/v4_choice.json` |
| `v4` | writes the chosen V4 submission and runs the official validator | ~15 min | `results/final/final_v4/output/matching_results.tsv` (+ copy in `submissions/final_v4/`) |
| `v5` | V5 features (`code/v5/extra_feats.py`), V2 recipe retrained on them, error accounting, test scores | ~3 h | `results/v5/recall_xgb_v5_s2.json`, `feats/test_scores_final_v5.parquet` |
| `loco5` | India → US with V2 features and with V5 features, pseudo-label check, unseen-country γ | ~3 h | `results/phase2/loco_India2US_v2.json`, `_v5.json` |
| `v5pl` | France pseudo-labels, stage-1 refit, France re-scored | ~1.5 h | `results/v5/pl_*.json` |
| `v5final` | both V5 submissions (per-country decoding), validator, pre-registered choice | ~20 min | `results/final/final_v5*/output/`, `results/v5/v5_choice.json` |
| `v6diag` … `v6pick` | **V6** (see `V6.md` §4 for the per-block table): density-matched universe, transferable features, stage 1, second cross-encoder, collective rounds, LOCO, test, plan, France pseudo-labels, validator + choice | ~11–14 h | `results/final/final_v6*/output/`, `results/v6/v6_choice.json` |

**V6** (V2, V3 and phase 2 already built): `bash run_all.sh --stage v6all`. **V5** is optional and superseded by V6 (`--stage v5all`); so is `v6a` (`--stage v6a`). Neither is part of `all` or `v6all`.

Existing submissions in `submissions/final_v2`, `final_v3`, etc. are **never overwritten**. The V4 file is only
copied into `submissions/final_v4/` if that folder doesn't have one yet.

## 5. The final V4 file

`run_all.sh` does this automatically. `select_v4.py` records the exact command in `results/phase2/v4_choice.json`
(`final_command`). It is one of:

```bash
# V4A corrected V3 (chain A replay: stage-1 half A -> XLM-R A -> GNN A, validation-style cosine)
python -u code/final/final_from_scores.py --test_scores p2/gnn/v4a_chainA_hash1_test_scores.parquet --col gnn --logit \
    --oof oof_gnn_mlx_v2.parquet --decode results/gnn/gnn_mlx_v2/decode_eval.json --out final_v4
# V4B simplified (chain A replay without XLM-R)
python -u code/final/final_from_scores.py --test_scores p2/gnn/v4b_gnnA_hash1_test_scores.parquet --col gnn --logit \
    --oof oof_gnn_v2.parquet --decode results/gnn/gnn_v2/decode_eval.json --out final_v4
# V4C V2-style (chain A replay of V2)
python -u code/final/final_from_scores.py --test_scores feats/p2_test_v4c_hash1_scores.parquet --col p \
    --oof oof_xgb_c2_blend_s2_v2.parquet --decode results/xgboost/xgb_c2_blend_s2_v2/decode_eval.json --out final_v4
```

Run these from the repository folder with `ER_NORM=v2 ER_CACHE=data/cache_v2 PYTHONPATH=code` set.
`run_all.sh` sets them for you. If the rule keeps V2, no V4 file is written and `submissions/final_v2` stays the
best robust submission.

Then validate the files:
`python data/utils/validate_submission.py --matching results/final/final_v4/output/matching_results.tsv --candidate results/final/final_v4/output/candidate_pairs.tsv --test-dir data/dataset/test`

## 6. What to send back (so the analysis can be finished)

Only small files are needed. Everything under `results/` except parquet/npy/pt:

```bash
git checkout -b results-phase2
git add results/phase2 results/audit results/final/*/test_stats.json results/_runs results/gnn results/xgboost results/neural_reranker
git commit -m "phase-2 results" && git push -u origin results-phase2
```

Or zip it: `zip -r phase2_results.zip results -x '*.parquet' '*.npy' '*.pt' '*.tsv'`.

## 7. Troubleshooting

| symptom | fix |
|---|---|
| `CUDA is not available` | wrong node / driver: `nvidia-smi` must list a GPU; install the cu121 torch wheel |
| out of GPU memory in a GNN step | add `--batch_s1 800` to that step's command (`bash run_all.sh --list`) and run it with `--only` |
| a step was killed half way | run the same command again (the half-written step has no `_done` marker) |
| `feats_train`/`feats_test`/`sim19` were killed half way | delete `data/cache_v2/feats/c2_train` (or `c2_test`, `c2_train_sim19`) and re-run |
| Hugging Face download blocked on the GPU node | pre-download on a login node with the same `HF_HOME` (section 2) |
| disk full | `data/cache_v2/p2/` and `data/cache_v2/emb/e3_test/paircos5_c2.parquet` can be deleted after phase 2; V6 needs ~150 GB free |
| `v6x_train` prints `WARNING: second cross-encoder failed` | allowed: V6 continues with XLM-R base; to retry: fix the cause, `rm data/cache_v2/_done/v6x_train` and re-run **before** `v6_c1` |

## 8. Folder map

```
run_all.sh             the single runner (this manual)
requirements.txt       pinned Python packages
code/common, blocking, embeddings, tfidf_fuzzy, xgboost, gnn, neural_reranker, final   V1-V3 pipeline (unchanged)
code/audit             25 Sep gap audit (gap_audit.py, test_decisions.py)
code/phase2            phase 2: cos_variants, gnn_lib/gnn_run, ablation, diff_accepts, loco_eval, v4, select_v4
code/phase2/tests      synthetic CPU smoke test: bash code/phase2/tests/smoke.sh
code/v5               V5 transferable features (used by V6), pseudo-label library
code/v6               V6: universe, collective rounds, prediction, decoding, pseudo-labels, choice; tests: bash code/v6/tests/smoke_v6.sh
results/               reports (json/csv/logs); results/phase2 = phase-2 outputs
submissions/           submitted matching files (final_v2 = 0.98332, final_v3 = 0.982814)
PHASE2.md              diagnosis, hypotheses, ablation design, V4 rule
V6_RESEARCH.md, V6.md  data study, literature, V6 design, decision rules, sources
data/                  created by the runner: dataset link, parquet caches (not in git)
```
