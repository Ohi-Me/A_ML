# Business Entity Resolution — reproducible pipeline (final submission: V6)

Links every Source-1 (S1) business to its Source-2 / Source-3 records and writes `matching_results.tsv` and
`candidate_pairs.tsv`. Everything is in `src/` (one runner, `src/run_all.sh`, and the Python package under
`src/code/`). No external data is used. Pretrained weights, fine-tuned here on the training pairs only:
{{MODEL_LIST}}. Every other model (TF-IDF blocking, n-gram bi-encoders, XGBoost) is trained from scratch on the
challenge training files.

## Environment
* One NVIDIA GPU with ≥ 40 GB memory (we used an H100), CUDA 12, Linux. ≥ 128 GB RAM and ~300 GB free disk
  recommended for the full run.
* Python 3.11:
  ```bash
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
* Internet access to huggingface.co, only for downloading the pretrained cross-encoder weights listed above.

## Reproduce the two output files (end to end, from the raw challenge data)

`DATASET` is the folder with `train/` (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`,
`train_ground_truth.tsv`) and `test/` (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).

```bash
cd src
bash reproduce_v6.sh /path/to/DATASET        # ~20-24 h on one H100; resumable (re-run the same command)
```

It runs these steps. The runner caches every step, logs to `src/results/_runs/<step>.log`, and skips steps that
already finished.
1. **V2 base.** Normalisation, hashed TF-IDF blocking on the GPU, n-gram bi-encoders (5 cross-fitted + 1
   all-folds), the C2 candidate union, out-of-fold pair cosines, 83 pair features and the stage-1/2 XGBoost models
   (`run_all.sh --stage v2`).
2. **XLM-R cross-encoder.** Fine-tuned on the uncertain pairs with cross-fitting, then scored on the test band
   (`mlx_prep`, `mlx_train`, `mlx_prep_te`, `mlx_score_te`).
3. **V6.**
   * Density-matched training universe (`v6_dms`).
   * Transferable features (`v6feat`).
   * Stage 1 at test density (`v6s1`).
   * Second cross-encoder (`v6x`).
   * Two collective sibling rounds (`v6col`).
   * Test chain (`v6test`).
4. **Decoding.** Isotonic calibration, per-source caps and expected-F0.5 per S1, with the settings the V6 plan
   fixed ({{FINAL_ARGS}}). Writes `src/results/final/final_v6/output/{matching_results,candidate_pairs}.tsv`.
5. **Validation.** The official validator runs on both files.

The full research chain (including the evidence behind every decision: ablations, leave-one-country-out, the
pre-registered selection) is `bash run_all.sh --data DATASET --stage all` (see `docs/V6.md`). `bash run_all.sh --list`
prints every step and its exact command.

GPU XGBoost (`hist`) and GPU fine-tuning are not bit-exact across runs or hardware. A rerun reproduces the
submission up to a small number of borderline pairs.

## Layout of `src/`
```
run_all.sh          the single runner (steps, caching, logs)
reproduce_v6.sh     the exact steps behind the submitted file
code/common         io, normalisation, transliteration, folds, pair features, decisions, expected-F0.5 decoder
code/blocking       hashed TF-IDF retrieval on the GPU, candidate union
code/embeddings     n-gram bi-encoders (cross-fitted), pair cosines
code/tfidf_fuzzy    pair feature runner
code/xgboost        test-prior simulation, XGBoost stages, decision evaluation
code/neural_reranker multilingual cross-encoders (prep_pairs.py, ml_cross_encoder.py)
code/v5             transferable features (ambiguity, name specificity, street core)
code/v6             density-matched universe, collective rounds, test chain, decoding, plan / choice
code/phase2, gnn, audit, final   diagnostics and earlier versions (V2-V4) used for the evidence
Data/student_resource/utils/validate_submission.py   the official validator
docs/               V6.md (design + decision rules), V6_RESEARCH.md (data study + literature), EXPERIMENT_LOG.md
```
