# Business Entity Resolution — code

Matches every Source-1 (S1) business to its Source-2 / Source-3 records. Output: `matching_results.tsv` and
`candidate_pairs.tsv`. Everything runs on one NVIDIA GPU (we used an H100 NVL MIG 3g.47gb slice, 46 GB) with at
most 32 GB of host RAM per stage. No external data is used. The only pretrained weights are
`FacebookAI/xlm-roberta-base` (MIT, 278M parameters, fetched from the Hugging Face hub), fine-tuned here on the
training pairs; every other model is trained from scratch on the challenge training files.

## Layout

```
src/
  common/      io, normalization (normalize.py), learned transliteration maps (translit.py), folds (split.py),
               pair features (features.py), stage-2 features, decision rules (decide.py, decode.py), metric
  blocking/    hashed TF-IDF retrieval on the GPU (sparse_tfidf.py), candidate union (build_candidates.py)
  embeddings/  n-gram bi-encoder (ngram_biencoder.py), 5 cross-fitted bi-encoders (oof_biencoder.py)
  tfidf_fuzzy/ pair feature runner (pair_features.py)
  xgboost/     test-prior simulation (simulate_drop.py), stage-1 model (train_xgb.py), stage-2 model (stage2.py),
               decision evaluation (decode_eval.py), error analysis
  neural_reranker/ multilingual cross-encoder for the uncertain pairs (ml_cross_encoder.py), pair texts (prep_pairs.py)
  gnn/         local edge GNN on the candidate graph (edge_gnn.py), OOF table (make_oof.py), evaluation
  final/       all-folds refit + test stage 1/2 (predict_test_v2.py), fusion (fuse.py), submission writer
               (final_from_scores.py), run_all_local.sh, validate.sh
```

## Run end to end

Put the challenge data at `data/dataset/{train,test}/*.tsv` and the validator at `data/utils/validate_submission.py`
(same layout as `student_resource/`), install `requirements.txt`, then from the folder that holds `src/` renamed to
`code/` (the scripts import `common.*` from `code/`):

```bash
bash code/final/run_all_local.sh v2
```

This runs, in order: normalization -> TF-IDF blocking -> bi-encoders (5 cross-fitted + 1 all-folds) -> candidate
pairs -> out-of-fold pair cosines -> pair features -> test-prior simulation -> stage-1 XGBoost (full + no-embedding,
blended) -> stage 2 -> all-folds refit and test stage-1 scores -> multilingual cross-encoder on uncertain pairs ->
edge GNN on the candidate graph -> calibration + expected-F0.5 decoding -> the two TSV files in
`results/final/final_v3/output/` -> official validator. Intermediate artifacts go to `data/cache_v2/` (about 60 GB).
Runtime on one MIG slice: about 6 hours, mostly TF-IDF retrieval (~50 min per split), feature building and the
cross-encoder fine-tuning (~35 min).

Randomness is fixed (fold hashing by entity id, fixed seeds for the bi-encoders, Monte Carlo decoding and the
simulated S1 drop), so reruns give the same files up to GPU floating point order.

## Validation protocol (short)

- Folds by S1 entity (`crc32(id) % 5`); fold 0 is validation. The search space is always all S1 and all S2/S3
  records, like the test.
- Anything learned (transliteration maps, bi-encoders, models, calibration, thresholds) is fitted on training folds
  only; thresholds / decoding settings are chosen on folds 1-2 and reported on fold 0.
- Test-prior simulation: 19% of S1 are removed so their records become "no-match" records, which gives the same
  records-per-S1 ratio as the test set (about 40% distractor records).
