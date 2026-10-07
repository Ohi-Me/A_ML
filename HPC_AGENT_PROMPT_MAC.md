# Run V7 check + V8 on the college H100 from a Mac (Claude Code / any coding agent)

## Part 1 - set-up on the Mac (you, not the agent, ~5 minutes)

**1. Put these three files in one folder** (for example `~/A_ML_run`):
* `HPC_AGENT_PROMPT_MAC.md` (this file)
* `A_ML_code_v8.zip`
* `v8_job.pbs`

**2. Be on the college network** (college Wi-Fi / LAN, or the college VPN).

**3. Allow network access.** macOS blocks apps from reaching college machines like `10.10.11.201` unless they are
allowed ("Operation not permitted"):
* System Settings → Privacy & Security → **Local Network** → switch ON for **Terminal** and for your agent app.
* In the agent app, allow network commands (turn off its sandbox, or approve `ssh` / `scp` when it asks).

**4. Create the SSH key - in the normal Terminal app, not inside the agent.** The agent never types passwords, so it
needs this key. The second command asks for the account password once: type it yourself.
```bash
ssh-keygen -t ed25519 -f ~/.ssh/hpc_ed25519 -N ""
ssh-copy-id -i ~/.ssh/hpc_ed25519.pub ai_25901334@10.10.11.201
ssh -i ~/.ssh/hpc_ed25519 -o BatchMode=yes ai_25901334@10.10.11.201 "echo KEY LOGIN OK"
```
The last line must print `KEY LOGIN OK` without asking for a password. If it prints "Operation not permitted" or
times out, go back to steps 2 and 3.

When the work is finished, remove the key again: delete its line (it ends with your Mac's user name) from
`~/.ssh/authorized_keys` on the cluster.

**5. Open the agent in that folder and paste everything between the lines below.**

---

You are helping run a machine-learning pipeline (Amazon ML Challenge 2026, business entity resolution) on the college
H100 cluster. This Mac is on the college network. Read this whole prompt before doing anything.

**Access.** Cluster: `ai_25901334@10.10.11.201`, PBS scheduler (`/opt/pbs/bin`), project folder
`/tmp/ai_25901334/AmazonML` (it already holds a ~70 GB cache in `data/cache_v2` and all earlier results). Log in only
with the SSH key: `ssh -i ~/.ssh/hpc_ed25519 -o BatchMode=yes ai_25901334@10.10.11.201 "<command>"` (same `-i` for
`scp`). Never ask for, type or store a password. First run
`ssh -i ~/.ssh/hpc_ed25519 -o BatchMode=yes ai_25901334@10.10.11.201 "echo KEY LOGIN OK"`. If it fails, stop and tell
me the exact error (don't try other logins).

**Rules.**
* Nothing heavy on the login node: only short commands (ls, cat, tail, df, unzip, qsub, qstat). All compute runs as a
  PBS job through `v8_job.pbs`.
* Never delete or change anything in `data/`, `results/` or `submissions/` on the cluster, and never touch
  `/tmp/ai_25901334/WidarH100` (another project).
* Never upload anything to the challenge leaderboard.
* Environment fixes only: don't change the pipeline's code, features, models or decision rules.
* Keep a log of every command you ran and what it printed in `RUN_LOG.md` in this folder.

**Step 1 - V7 status and results.**
1. `export PATH=/opt/pbs/bin:$PATH; qstat -u ai_25901334`, `df -h /`, and find the V7 run:
   `find /tmp/ai_25901334 -maxdepth 6 -path '*final_v7/output/matching_results.tsv' 2>/dev/null` and
   `find /tmp/ai_25901334 -maxdepth 3 -name 'run_v7.out' 2>/dev/null`; show the last 30 lines of `run_v7.out`.
2. Copy to `./results_from_hpc/v7/` on this Mac: `results/final/final_v7/output/matching_results.tsv`,
   `results/final/final_v7/test_stats.json`, the whole `results/v7/` folder, and `run_v7.out`.
3. Report: is V7 finished, still running, or failed (and at which step)? What does `results/v7/v7_choice.json` say
   (`submit_order`, `dms_min_f0_f4`, `ceiling`)?

**Step 2 - start V8 (only if V7 is not running and `df -h /` shows >= 60 GB available; otherwise stop and report).**
1. `scp -i ~/.ssh/hpc_ed25519 A_ML_code_v8.zip v8_job.pbs ai_25901334@10.10.11.201:/tmp/ai_25901334/AmazonML/`
2. On the cluster, in that folder: `unzip -o A_ML_code_v8.zip` (it only updates code and docs), check
   `grep -c v8all run_all.sh` prints a number > 0, `mkdir -p results/_jobs`, then `qsub v8_job.pbs`. Note the job id.
3. Read `V8.md` and `AGENT_RUN_V8_PROMPT.md` in the unzipped folder: they explain the steps and what to watch.

**Step 3 - watch the job and save results.** Check every 20-30 minutes:
`qstat -u ai_25901334` and `tail -n 30 /tmp/ai_25901334/AmazonML/results/_jobs/v8all.log`.
* If the job is no longer in `qstat` and the log's last line is `EXIT rc=0`, V8 is done. Go to the final copy.
* If it disappeared with another exit code, or was killed (walltime, or the cluster's kills at ~:44 of even hours):
  look at `results/_runs/<last step>.log`. If it's an environment or kill problem, run `qsub v8_job.pbs` again
  (finished steps are skipped). If the same step fails twice with a code error, stop and report the log.
* If `results/v8/sib_probe.json` shows `id_check.spearman_true_pairs` far from `spearman_random_pairs`
  (difference > 0.05), cancel the job with `qdel <id>` and report: that needs the owner's decision.
* After `v8_miss_probe` and again after `v8_prune_test` finish, copy `results/v8/`, `results/v7/miss_probe.json` and
  `results/v7/cands_c3w*_*.json` to `./results_from_hpc/v8/`.

**Final copy (when V8 is done)** to `./results_from_hpc/v8/`: the whole `results/v8/` folder,
`results/final/final_v8/output/matching_results.tsv`, `results/final/final_v8/test_stats.json`,
`results/xgboost/xgb_v8_c1/decode_eval.json`, `results/xgboost/xgb_v8_c2/decode_eval.json`,
`results/_runs/v8_val.log`, `results/_jobs/v8all.log`.
Then write `./results_from_hpc/SUMMARY.md`: V7 status; V8's `v8_choice.json` (`submit_order`, `first_choice_file`,
`dms_min_f0_f4`, `ceiling`, `v8_better`, `v8_consistent`); the pruner numbers (`pairs_per_record_before/after`,
`by_fold.0.kept_share_of_true`); and the validator result. Zip `results_from_hpc` as `results_from_hpc.zip`.

---
