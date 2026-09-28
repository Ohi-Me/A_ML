# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** {{TEAM}}  
**Team Members:** {{MEMBERS}}  
**Submission Date:** {{DATE}}

---

## 1. Executive Summary
We generate candidates with GPU TF-IDF retrieval plus learned n-gram bi-encoders. A two-stage XGBoost pair model is
enriched with two multilingual cross-encoders and "collective" evidence from the other records of the same Source-1
entity. The final decision is an expected-F0.5 choice per Source-1 entity. The key innovation is a **density-matched
training universe**: the test index is half as dense as the training one (US 0.66 M vs 1.32 M entities). Every model
is trained, calibrated and selected on training data rebuilt at the test's density. This removed the over-acceptance
on test that capped our earlier versions. Leaderboard: {{LB}}.

---

## 2. Methodology

### 2.1 Problem Analysis
* Source 1 is duplicate-free; every S2/S3 record matches at most one S1. There are 0 cross-country true pairs, so
  blocking is done per country. 5.6 % of S1 have no match. **No S1 has more than 5 S2 or 6 S3 records** (used as a
  hard constraint in decoding).
* Noise operators measured on 3,000 true pairs:
  * Names: key equal only 59 %; dropped or added words, one-token typos, native scripts (7 %: Devanagari, Telugu, …),
    domain forms, accent noise ("Gároa"), bracketed legal forms, honorifics.
  * Addresses: house number changed in 15 %, missing in 10 %, address empty in 5 %; S2 upper case with state codes,
    S3 title case with full state names; France: "N°", bis/ter, street-type abbreviations, department vs region.
* **Hierarchical generator:** base entity → one version per source → per-record noise. Same-source records of one
  entity share 81 % identical addresses (record vs S1: 76 %), and a changed house number is usually shared by all
  of a source's records. A hard record can be linked through an easy sibling.
* **Density shift:** the test index is half as dense as training for the US (92 % for India; France is unseen).
  Competition and count features shift with density. Models trained at training density over-accepted on test US,
  and the leaderboard order of our versions followed exactly this over-acceptance.
* Most of the loss is missed matches, not false merges (precision 99.85 %, recall 97.7 % on validation). More than
  half of the misses involve empty-address records whose name is shared by several S1.

### 2.2 Solution Strategy
**Approach Type:** Blocking + classifier (two-stage gradient boosting with stacked cross-encoders and collective
features) + expected-F0.5 decoding.  
**Core Innovation:** training, calibrating and selecting at the test's index density (density-matched universe).
Also: collective sibling evidence exploiting the per-source generator, and a decoder that optimises the expected
macro F0.5 per S1 under the ground-truth per-source caps.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** four ranked retrieval lists per record, all within the record's country:
  * TF-IDF name cosine (char 3-grams + words, hashed, GPU sparse matmul)
  * TF-IDF address cosine
  * combined name+address TF-IDF
  * a learned n-gram bi-encoder (256-d, InfoNCE with same-state hard negatives; five cross-fitted models so that
    training features are out-of-sample)
  
  Candidates = union of the top 5 (bi-encoder), 3 (combined), 2 (name) and 2 (address). Each rank is kept as a
  feature (99 = not in that list).
- **Candidate pairs generated:** {{CAND_PAIRS}} test pairs ({{CAND_PER_S1}} per S1 on average); `output/candidate_pairs.tsv`.
- **How you ensured true matches were not lost:**
  * Four complementary lists (lexical name, lexical address, combined, learned semantic).
  * Normalisation before retrieval: transliteration of native scripts, legal-form families, street-type and state
    maps.
  * Measured on validation: candidate recall {{CAND_RECALL}} of all true pairs. The best any scorer can reach on
    these candidates at test density is F0.5 {{CEILING}}.

---

## 4. Matching Model

**Features used ({{N_FEATURES}} in the final model):**
- Name features: TF-IDF cosines; bi-encoder cosine; Jaro-Winkler, token-set and partial ratios; token alignment;
  name-key equality; legal-form family match or conflict; domain and native-script flags; name specificity
  (document frequency of shared tokens among the country's S1).
- Address features: TF-IDF cosine; first house number equal / missing / conflicting; number-set overlap; street-core
  (address without numbers) Jaccard and token-set ratio; state match; empty-address flags.
- Other:
  * Retrieval ranks, and gaps and ranks against the record's other candidates and the S1's other records.
  * Record ambiguity (how many S1 look equally good).
  * Stage-1 probability context.
  * Sibling similarity; collective evidence from the S1's other likely records (same / cross source: identical
    address, shared house number, shared non-S1 number, identical name key, support sum, per-source counts, margin).
  * Scores of two multilingual cross-encoders on the uncertain pairs: {{MODEL_LIST}}.

**Model type:** XGBoost (GPU):
* Stage 1: blend of a full model and a model without embedding features.
* Two collective rounds: round 2 recomputes the collective evidence from round-1 probabilities.

All stacked inputs are out-of-fold (5 folds by S1; two cross-fitted halves).  
**Threshold selection method:** No fixed threshold.
* Isotonic calibration on held-out folds at test density.
* Each record goes to its best S1.
* Per-source caps (≤ 5 S2, ≤ 6 S3).
* Per S1, choose the number of records that maximises the **expected F0.5** (Monte Carlo over calibrated
  probabilities); the decoder settings are tuned on folds 1-2.
* Every choice (caps, prior correction, unseen-country setting, final system) was fixed by a pre-registered rule
  before seeing its result.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation {{VAL_F05}} (held-out folds 0 / 4 of the density-matched universe; precision
  {{VAL_P}}, recall {{VAL_R}}). Leaderboard {{LB}}.

  Progression on validation (the V6 figure is at test density, the others at training density):

  | version | validation F0.5 |
  |---|---|
  | exact-key rules | 0.641 |
  | first learned model | 0.983 |
  | V2 | 0.987 |
  | V3 (GNN + cross-encoder) | 0.991 |
  | V6 (at test density) | {{VAL_F05}} |

  Leaderboard: V2 0.98332 → V3 0.98281 → V6 {{LB}}. V3's validation gain did not transfer; V6 fixed that with the
  density-matched training.
- **Common false positives (wrong merges):**
  * "Twin" distractor records that copy an S1's name and address almost exactly but belong to no S1.
  * Same-name businesses in the same city with slightly different addresses.
- **Common false negatives (missed matches):**
  * Records with an empty address whose name is shared by several S1 (content-ambiguous; the decoder abstains).
  * Records whose name was replaced by a DBA / new name.
  * Changed house numbers.
  * Native-script names with a truncated address.
  * Blocking misses, about 1.2 % of true pairs, mostly empty-address records.

---

## 6. Conclusion
The largest single gain did not come from a bigger model. It came from making training look like test: after
simulating the test's index density, validation and leaderboard finally agreed. Collective evidence from sibling
records and multilingual cross-encoders added recall at unchanged precision. The remaining gap is candidate recall
and the unseen country (France). Our pre-registered, label-free selection rules kept every submission choice free of
test-label feedback.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`:
* `src/run_all.sh` is the single runner: every step with caching and logs.
* `src/reproduce_v6.sh DATASET` runs exactly the steps behind the submitted file and writes
  `src/results/final/final_v6/output/{matching_results,candidate_pairs}.tsv`, then runs the official validator.
* `README.md` gives the environment and reproduction steps; `requirements.txt` has the pinned versions.

Main entry points under `src/code/`:

| step | scripts |
|---|---|
| normalisation | `common/prep.py` |
| blocking | `blocking/sparse_tfidf.py`, `embeddings/oof_biencoder.py`, `embeddings/ngram_biencoder.py`, `blocking/build_candidates.py` |
| features | `tfidf_fuzzy/pair_features.py`, `v5/extra_feats.py` |
| density universe | `v6/simulate_universe.py` |
| stage 1 | `xgboost/train_xgb.py`, `xgboost/blend_oof.py` |
| cross-encoders | `neural_reranker/prep_pairs.py`, `neural_reranker/ml_cross_encoder.py` |
| collective rounds | `v6/train_collective.py` |
| test chain | `v6/predict_v6.py` |
| decoding | `v6/final_v6.py` |

**Compliance:**
* Only the provided training/test files are used.
* Test labels are never used.
* No entity IDs or row order are used as signals.
* Pretrained models: {{MODEL_LIST}}, all MIT / Apache-2.0 and below 8 B parameters.
* XGBoost and PyTorch are Apache-2.0 / BSD.

### B. Additional Results
Test decisions per country, from `results/final/final_v6/test_stats.json`:

{{TEST_TABLE}}

Validation per country (density-matched universe, fold 0):

{{VAL_TABLE}}

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
