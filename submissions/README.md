# Submission files

One folder per system. `matching_results.tsv` is what was (or could be) uploaded to the leaderboard. Validation numbers
are from `EXPERIMENT_LOG.md` (the older systems on SIM19 validation, V6 / V8 on the density-matched DMS validation).

| folder | system | leaderboard | notes |
|---|---|---|---|
| `final_c1` | V1 (C1 candidates, XGBoost) | - | validation 0.983 (plain) |
| `final_c2v1` | V1.5 (C2 candidates, OOF embeddings) | - | validation 0.9848 |
| `final_v2` | V2 (normalisation v2, stage 2) | 0.98332 | validation 0.9870 |
| `final_v3` | V3 (GNN + XLM-R cross-encoder) | 0.982814 | validation 0.9908 on SIM19, which overstated test |
| `final_v3nognn` | V3 without the GNN | - | DMS 0.98981 |
| `final_v4` | V4A (chain replay) | 0.981 | |
| `final_v6` | **V6** (density-matched training, 2 cross-encoders, collective rounds) | 0.988109 | DMS 0.99127. This is the system documented in `Enigma_submission/` |
| `final_M1_consensus_add` | V6 + 12.7 k pairs that 6 of 8 systems agree on | 0.988059 | |
| `final_T1_trim` | V6 minus 4.3 k pairs no other system predicts | 0.988285 | best leaderboard score so far |
| `final_v8` | V8 (wide candidates + pruner) | not uploaded | DMS 0.98787 (below V6); ceiling 0.99759 |

`final_v6/candidate_pairs.tsv` (1.09 GB, kept locally, not in git) is the candidate set shared by V2, V3, V3-no-GNN and
V6 (all copies had the same md5 `b5db23a2...`). `final_v8/candidate_pairs.tsv` is V8's own pruned set (277 MB, local only).

`Enigma_submission.zip` is the final package for the challenge portal (V6 files, source code, documentation) with
`output/`, `code/` and `Documentation_template.md` at the zip root, as the rules require.
`Enigma_submission_wrapped_in_folder.zip` has the same files inside one extra top-level `Enigma_submission/` folder
(what Windows Explorer produces). Large files (TSVs over 100 MB, zips) are not in git: only the matching files up to
100 MB are tracked.
