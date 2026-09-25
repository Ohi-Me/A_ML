# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

We clean names and addresses (with a transliteration map learned from the training pairs), find candidate S1
entities for every S2/S3 record with two GPU retrievers (hashed TF-IDF and our own n-gram bi-encoder), and score the
candidates with gradient-boosted pair models. On top of that, a local graph neural network on the candidate graph
and a multilingual cross-encoder (XLM-RoBERTa base, MIT) for the uncertain pairs re-score each pair in context, and a
per-entity decoder picks the match set with the highest expected F0.5. Everything is trained on the H100 from the
training files only. On a validation built to look like the test set, macro F0.5 is **0.9908**.

---

## 2. Methodology

### 2.1 Problem Analysis

- Train: 2.21M S1 against 10.3M S2/S3 records; test: 1.73M S1 against 9.97M records. The test adds France (15% of
  S1), which has no labels in training. The country label is treated as an open set everywhere.
- Every S2/S3 record belongs to at most one S1; true pairs always share the country label.
- 5.6% of training S1 have no match; most have 2 to 6 (up to 11).
- **Train/test prior shift.** 26% of training S2/S3 records match no S1. The test has 5.76 records per S1 against
  4.68 in train while predicted matches per S1 stay ~3.35, so about 40% of test records match no S1. These records
  cause most false merges, so we validate (and train) under a simulation of this prior (section 4).
- Only 11% of true pairs share the lower-case name. Name noise: honorific / symbol prefixes, legal-suffix swaps and
  duplication, word shuffles, typos and digit swaps (Y0uth, 6lobal), domain forms, appended IDs and phones, generic
  word swaps (Academy -> Center), whole names in Indic scripts, and completely different trade names at the same
  address. Address noise: reordered parts, abbreviations, state as code / name / native script, house numbers
  changed or split (C-9-7 vs C-97, 105 vs 05), PMB / Unit added, French region vs department, ~3% empty addresses.
- Among confident candidate pairs, 0.17% are "records of no S1 that look exactly like an S1" (duplicates in the
  source data). These are mostly irreducible.

### 2.2 Solution Strategy

**Approach Type:** Blocking (TF-IDF + learned bi-encoder, GPU) -> XGBoost pair model -> local GNN + multilingual
cross-encoder -> calibrated expected-F0.5 decoding.
**Core Innovation:** (1) cross-fitted everything (bi-encoders, pair models, GNN, cross-encoder) so the features and
scores seen in training look like the ones at test time; (2) a test-prior simulation (19% of S1 removed) used for
validation, calibration and training; (3) record-level competition learned by a GNN over the candidate graph;
(4) per-entity expected-F0.5 decoding.

```
raw S1/S2/S3 -> normalization v2 (learned transliteration, legal forms, address words, states / French regions)
  -> retrieval per country: hashed TF-IDF (name, address, both)  +  n-gram bi-encoder (5 cross-fitted models)
  -> candidate union (84.9M train / 83.1M test pairs, recall 98.8%)
  -> ~90 pair features -> XGBoost stage 1 (full + no-embedding twin, blended)
  -> uncertain pairs (0.02 <= p <= 0.98, ~1.3%) -> XLM-RoBERTa cross-encoder
  -> local edge GNN on the candidate graph (stage-1 score + key features + cross-encoder score)
  -> isotonic calibration -> expected-F0.5 decoding per S1 -> matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

Blocking runs from the record side: every S2/S3 record looks for its S1 among the S1 of the same country label.

- **Normalization (v2).** Names: Indic-script words replaced through a map learned from training pairs (word-by-word
  alignment of transliterated names, fitted on training folds only; unseen words fall back to `anyascii`),
  honorifics and symbol prefixes removed, legal forms mapped to one token (private/pvt, limited/ltd, l.l.c./llc,
  sas/sarl/eurl ...), `&` -> and, digit swaps inside words fixed, domain names split. Addresses: street / Indian /
  French address words to one form, ordinals and number words to digits; states (codes, full names, native-script
  names through a learned map, French regions and departments) moved to a separate state field.
- **Retriever 1 - hashed TF-IDF on the GPU.** Name = character 3-grams + words, address = tokens, 2^19 hashed
  buckets per field, sublinear tf; IDF is a label-free statistic of the corpus being searched. Batches of 2,048
  records are scored against all S1 of their country with one cuSPARSE sparse x dense product; top-k on a contiguous
  fp16 copy (5x faster than on the strided dimension). Lists: name, address, name+address.
- **Retriever 2 - n-gram bi-encoder (trained from scratch).** EmbeddingBag over the same hashed n-grams (weighted by
  TF-IDF) -> MLP -> 256-d unit vector; symmetric InfoNCE on true pairs with in-batch negatives drawn from the same
  country and state. Five cross-fitted models: each training record is embedded by the model that never saw its
  entity; test uses a model trained on all folds. Retrieval = fp16 matrix product + top-k (~30 us per record).
- **Candidate set** = union of bi-encoder top 5, TF-IDF name+address top 3, name top 2, address top 2:
  84.9M training pairs (8.2 per record), 83.1M test pairs. **Recall of true pairs 98.8%** (validation).
  Most of the missed 1.2% are records with an empty address whose name is shared by several S1 businesses.
- `candidate_pairs.tsv` = exactly this candidate table grouped by S1; every pair in it is scored by the models and
  every predicted match comes from it.

---

## 4. Matching Model

**Features (~90, country-generic comparisons only):**
- Retrieval: ranks in each list, TF-IDF name / address cosines, out-of-fold bi-encoder cosine.
- Competition: for each record, gap to its best and runner-up S1 and its rank; for each S1, gap to its best record,
  rank, number of candidates; how common the name key is among S1 and records.
- Name: rapidfuzz ratio / token sort / token set / partial / Jaro-Winkler on the core name, on the name without
  generic words and without spaces; token Jaccard and containment both ways; first token; number of tokens with no
  fuzzy partner on each side; legal form present on each side and legal-family relation (same / different / added /
  dropped); domain and native-script flags.
- Address: token set / sort / partial ratios, token Jaccard and containment, house number equal / edit distance /
  relative difference, numbers compared per token (C-9-7 = C-97) with overlap and conflicts, state comparison,
  empty address.

**Models (all on the H100, all cross-fitted in two halves: folds {1,2} score folds {0,3,4}, folds {3,4} score
{1,2}; for the test set the stage-1 XGBoost models are refitted on all folds, while the cross-encoder and the GNN
use the mean of their two half-models):**
1. **Stage 1 - XGBoost** (CUDA hist, depth 9, eta 0.06, early stopping): a model with all features and a twin
   without embedding features, blended 0.75 / 0.25 for countries with training labels and 0.5 / 0.5 for countries
   without (France): the bi-encoder never saw French text, and the blend costs nothing on validation.
2. **Multilingual cross-encoder** for the uncertain pairs: XLM-RoBERTa base (MIT, 278M parameters) fine-tuned on raw
   text (native scripts and accents kept), "S1 name | address" vs "record name | address", 2 epochs, bf16.
3. **Local GNN** on the candidate graph: nodes = S1 and records, one edge per candidate pair carrying the blended
   stage-1 logit, 24 key features and the cross-encoder logit. Two rounds of mean/max pooling over each S1 and each
   record update every edge, so a pair sees the record's competing S1 and the S1's other records. Trained on S1
   mini-batches with their 2-hop neighbourhoods built on the GPU.

**Threshold selection:** isotonic calibration of the GNN probability (fitted on folds 1-2), then for each S1 we
choose the number k of its best records that maximizes the expected F0.5, E[1.25 TP / (0.25 T + k)] (k = 0 scores
1 only when the S1 has no match), by Monte Carlo on the GPU. A threshold + runner-up margin rule is also tuned; the
rule with the better score on folds 1-2 is used (here expected-F0.5).

**Validation:** entity-level folds (crc32 of the S1 id), fold 0 held out; the search space is always all S1 and all
records. Test-prior simulation: 19% of S1 removed by hash, their records become no-match records (40.5% of records,
as in test), every candidate-dependent feature recomputed. Everything learned uses training folds only.

---

## 5. Results & Error Analysis

Macro F0.5 on the validation fold under the test-prior simulation (357,966 S1):

| System | F0.5 | pair P | pair R | FP | FN |
|---|---|---|---|---|---|
| Exact normalized keys (baseline, plain validation) | 0.641 | 0.973 | 0.434 | | |
| Stage 1 XGBoost, first features (C1) | 0.9818 | 0.9958 | 0.9565 | 4,982 | 53,883 |
| + cross-fitted embeddings, number / legal / token features (C2) | 0.9852 | 0.9962 | 0.9644 | 4,566 | 44,162 |
| + stage 2 context model (v2 normalization, blended stage 1) | 0.9869 | 0.9968 | 0.9696 | 3,915 | 37,615 |
| GNN on the candidate graph (replaces stage 2) | 0.9884 | 0.9978 | 0.9713 | 2,673 | 35,518 |
| **GNN + multilingual cross-encoder + expected-F0.5 decoding (final)** | **0.9908** | **0.9987** | **0.9744** | **1,629** | **31,722** |

Candidate recall ceiling 98.8%. Singleton accuracy 0.992.

- **Common false positives:** records that belong to no S1 but copy an S1's name and address almost exactly
  (duplicates in the source data; confident errors are only ~0.1% of confident pairs), and empty-address records
  whose name exists at several S1.
- **Common false negatives:** blocking misses of empty-address records with a shared name (nearly half of FN), and true
  pairs with corrupted house numbers plus a changed name where the evidence stays ambiguous.
- Ablations: embedding features +0.08; F2 number/legal/token features 33% of stage-1 gain; deeper trees no gain;
  from-scratch char cross-encoder +0.01 (weak, AUC 0.66); pretrained multilingual cross-encoder AUC 0.945 vs 0.916 on
  uncertain pairs.

---

## 6. Conclusion

The biggest wins came from understanding the data rather than from bigger models: learned transliteration and
number-aware address features, cross-fitting so that features look the same at train and test time, and a
validation that copies the test's higher share of no-match records. On that base, a small GNN over the candidate
graph and a pretrained multilingual cross-encoder on the ~1% uncertain pairs added +0.4 points of macro F0.5.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` (entry point `final/run_all_local.sh`, see README.md):
`common/` (normalization, transliteration, features, decoding, metric), `blocking/` (GPU TF-IDF retrieval,
candidate union), `embeddings/` (bi-encoders), `tfidf_fuzzy/` (pair features), `xgboost/` (simulation, stage 1,
stage 2, decoding evaluation, error analysis), `neural_reranker/` (cross-encoders), `gnn/` (edge GNN), `final/`
(test prediction, fusion, submission writer, validator wrapper). `EXPERIMENT_LOG.md` lists every experiment.

### B. Additional Results

Blocking recall@k (validation): TF-IDF name+address @1 0.972 / @5 0.986 / @20 0.991; bi-encoder @1 0.968 /
@5 0.984 / @20 0.991. Normalization v2 vs v1 on France test candidates: state recognized 0% -> 64% (mean), top-
candidate address token-set similarity 88.3 -> 93.4. Decision-rule comparison, per-country breakdowns and error
samples are in `results/` of the code archive.
