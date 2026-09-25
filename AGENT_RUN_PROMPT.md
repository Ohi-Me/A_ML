# Prompt for a coding agent (Antigravity or similar) that runs the full pipeline

Copy everything between the lines into the agent.

---

You are running an existing, finished machine-learning pipeline end to end on this machine. Your job is to execute
it, monitor it, fix only environment/operational problems, and report the results. Do not redesign the models.

## Project
Repository: this folder (GitHub `Ohi-Me/A_ML`, branch `claude/dazzling-gauss-3jgxpd`). It is a business entity
resolution pipeline (Amazon ML Challenge 2026). Read these files first, in this order:
1. `H100_MANUAL.md`: how to run everything (setup, commands, times, outputs, troubleshooting)
2. `PHASE2.md`: what the phase-2 experiments test and how V4 is chosen
3. `run_all.sh --list` output: every step and its exact command

## Data
The challenge data has been added to the repository. Find the folder that contains `train/` and `test/` with these
files: `train_source1.tsv train_source2.tsv train_source3.tsv train_ground_truth.tsv test_source1.tsv
test_source2.tsv test_source3.tsv` (most likely `Data/student_resource/dataset/`). Use that folder as `--data`.
Do not modify, move or re-encode the data files.

## Requirements check (stop and tell me if any fails)
- Linux (or WSL2 on Windows) with bash. `run_all.sh` is a bash script.
- An NVIDIA GPU with at least 40 GB memory and a CUDA 12 driver: `nvidia-smi` must list it. If there is no such GPU
  on this machine, STOP and tell me. Do not try to run on CPU, and do not reduce the data to make it fit.
- At least 250 GB of free disk in the repository folder (`df -h .`).
- Internet access once, to download `FacebookAI/xlm-roberta-base` from Hugging Face (or a pre-filled `HF_HOME`).

## Setup
```bash
conda create -y -n er python=3.10 && conda activate er     # or python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash run_all.sh --data <DATASET_FOLDER> --dry               # must print the GPU and the planned steps
```

## Run
```bash
nohup bash run_all.sh --data <DATASET_FOLDER> > run_all.out 2>&1 &
tail -f run_all.out
```
The blocks run in this order: v2 → v3 → audit → phase2 → loco → select → v4. The total is roughly 8–12 hours. Each
step's log is in `results/_runs/<step>.log`. Check progress periodically and don't restart steps that are running.

## If a step fails
- Read `results/_runs/<step>.log` and the troubleshooting table in `H100_MANUAL.md` section 7.
- Allowed fixes: install missing packages, free disk, set `HF_HOME`, lower `--batch_s1` for a GNN step that runs
  out of GPU memory (run that step alone with `bash run_all.sh --only <step>` after changing it), delete a
  half-written feature folder as the manual says, then re-run `bash run_all.sh --data <DATASET_FOLDER>` (it resumes).
- NOT allowed: changing model code, features, hyperparameters, thresholds, fold definitions or the V4 selection rule
  in `code/phase2/select_v4.py`; skipping steps; editing result files by hand; using external data or APIs.
- If a failure needs a code change beyond the above, stop and report the exact error with the last 50 log lines.

## Hard rules
- Never overwrite or delete anything in `submissions/` (V2 and V3 are the reference submissions).
- Do not upload anything to the competition leaderboard.
- Do not commit large files: no `data/`, `*.parquet`, `*.npy`, `*.pt`, `hf_cache/`, or any `*.tsv` over 50 MB.

## When it finishes, report back
1. Whether every step succeeded (`ls data/cache_v2/_done/`) and the total run time.
2. `results/phase2/ablation.csv`: the full table, especially val_f0, val_f4, test_matches, delta_vs_V2,
   extra_removed and inflation_US / inflation_India for every system.
3. `results/phase2/cos/cos_report.json`: the `agreement` section (correlation of the two cosine computations,
   between-model spread) and the KS values of test_mean5 vs test_hash1 against validation for US and India.
4. `results/phase2/diff_accepts.json`: the top 5 variables of `divergence_val_vs_test` for US, and
   `test_V3_only_profile`.
5. `results/phase2/loco_India2US_noemb.json` and `loco_India2US.json`: the `component_deltas_iso_ef` section.
6. `results/phase2/v4_choice.json`: the chosen system, the reasons other candidates were rejected, and
   `final_command`.
7. The validator result for `results/final/final_v4/output/` (PASS or the issues), and the
   `results/final/final_v4/test_stats.json` numbers (matches, empty S1, per-country).
8. Also the replay checks at the top of `ablation.json` (`replay_check_V2`, `replay_check_V3`): how closely the
   rebuilt V2/V3 match the submitted files.

Then save the small result files to GitHub on a new branch:
```bash
git checkout -b results-phase2
git add results/phase2 results/audit results/final/*/test_stats.json results/_runs results/gnn results/xgboost results/neural_reranker
git reset -q -- '*.parquet' '*.npy' '*.pt'
git commit -m "Phase-2 run results" && git push -u origin results-phase2
```
Keep `results/final/final_v4/output/matching_results.tsv` locally; it is the candidate V4 submission. Do not submit it.

---
