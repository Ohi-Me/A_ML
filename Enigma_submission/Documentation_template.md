# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Enigma
**Team Members:** Rohit Kumar, Vishesh Shekhawat, Sriyansh, Veeky Kumar
**Submission Date:** September 28, 2026

---

## 1. Executive Summary

We find candidate S1 entities for every Source 2 / Source 3 record with two GPU retrievers (a hashed TF-IDF index and
our own n-gram bi-encoder), score the candidates with XGBoost on ~100 name / address / competition features, re-score
the uncertain pairs with two multilingual cross-encoders (XLM-RoBERTa base and bge-reranker-v2-m3), and let the S1's
other records vote in two "collective" rounds. A calibrated decoder then picks, for every S1, the set of records with
the highest expected F0.5. The key idea of the final version (V6) is that everything is trained and tuned on a
validation set rebuilt to look like the test set, including how crowded it is. Validation macro F0.5: **0.9914**.

---

## 2. Methodology

### 2.1 Problem Analysis

What we learned from the data before building anything:

| finding | example / number | what we did about it |
|---|---|---|
| Records in Indic scripts | `राम मार्केटिंग प्राइवेट लिमिटेड` is the S2 record of "Ram Marketing Pvt Ltd"; ~10 % of S2/S3 names are in Devanagari, Telugu, Tamil, ... while S1 is always Latin | a word-by-word transliteration map **learned from the training pairs** (training folds only), `anyascii` as fallback |
| Name noise | `-- Holloway Peak Inc Seafood`, `wilfordhancock.com`, `Y0uth` for Youth, `lnc` for Inc, legal forms swapped or doubled (Pvt / Private / Ltd / Limited / LLC / SARL) | strip prefix symbols and honorifics, split domains, fix digit/letter swaps, map every legal form to one token, keep a "key" without generic words |
| Address noise | `105 ELM ST` vs `105 Elm Street`, parts reordered, state as code / name / native script, house number changed (`C-9-7` vs `C-97`), ~3 % empty addresses | address word canonicalisation, states moved to their own field, numbers compared token by token |
| Unseen country | France appears only in test: `63 R. DE DIEPPE, LILLE, Hauts-de-France` | country treated as an open label; French regions and departments recognised as states; no feature uses the country's name |
| Many "distractor" records | 26 % of training S2/S3 records match no S1, but ~40 % of test records do (test has 5.8 records per S1, train 4.7, while matches per S1 stay ~3.4) | validation rebuilt with the test's share of distractors |
| **Test is less crowded** | test US has 0.66 M S1, training US 1.32 M; India 0.81 M vs 0.88 M | the V6 key idea: train and tune on a universe with the **test's density** (§2.2) |
| Hierarchical data | records of one entity from the same source share details the S1 does not have (same address in 81 % of same-source sibling pairs) | "collective" features: a hard record can be linked through an easy sibling |
| Hard limits | no S1 has more than 5 S2 or 6 S3 records | the decoder enforces these caps |
| Metric | macro F0.5 per S1, singletons included: a wrong match costs about 3x what a missed match costs | calibrated probabilities + expected-F0.5 decoding per S1 |

### 2.2 Solution Strategy

**Approach Type:** Hybrid: multi-retriever blocking + gradient-boosted pair classifier + multilingual cross-encoders +
collective (sibling) re-scoring + expected-F0.5 decoding.

**Core Innovation:** a *density-matched* training and validation universe. Our earlier versions looked better on
validation than on the test set, and the cause turned out to be crowding: in the test's US data every business has
half as many look-alike competitors as in training, so a model tuned on training-density data accepts too much on
test. V6 rebuilds the training data per country at the test's density (it keeps 62 % of US entities, then removes 19 %
of S1 so their records become distractors, as in test) and trains, calibrates and tunes everything there.

**How we got there (one version at a time):**

| version | what we added | why | validation F0.5 |
|---|---|---|---|
| V1 | normalisation v1, hashed TF-IDF + n-gram bi-encoder blocking, 69 pair features, XGBoost | a fast, strong first system; exact keys alone found < 50 % of matches (F0.5 0.641) | 0.9818 |
| V1.5 | out-of-fold bi-encoders; number / legal-form / token-alignment features | the bi-encoder had seen the validation entities during training, which made its cosine look too good; house-number conflicts were the top error | 0.9852 |
| V2 | normalisation v2 (French regions / departments), a no-embedding model blended in (hedge for France), stage-2 context model | the bi-encoder never saw French text; context of the record's other candidates helps | 0.9870 |
| V3 | XLM-RoBERTa cross-encoder on uncertain pairs + a GNN on the candidate graph | the cross-encoder reads native scripts and accents: AUC 0.945 vs 0.916 on the hard pairs | 0.9908 |
| V4 | diagnosis of why V3 did better on validation than on test | the gap came from the test's lower density (US accepted 3 % more on test than on validation; India matched) | — |
| V5 | transferable features (record ambiguity, name specificity) and an unseen-country (India → US) check | make the model depend less on country-specific statistics | folded into V6 |
| **V6** | **density-matched universe**, second cross-encoder (bge-reranker-v2-m3), two collective rounds, per-source caps; GNN removed | fixes the validation / test gap and adds recall through sibling evidence | **0.9914** |

V1–V3 were validated on our earlier validation set (test-like distractor share, training density), V6 on the
density-matched one, which is harder and much closer to test. Re-scored on the density-matched validation, the V2 chain
gets 0.9868 and the V3-without-GNN chain 0.9898, against 0.9913 for V6 (§5).

---

## 3. Candidate Generation (Blocking)

Blocking runs from the record side: each S2/S3 record looks for its S1 among the S1 with the same country label
(true pairs always share it, so this loses nothing).

- **Blocking keys used:**
  1. **Hashed TF-IDF on the GPU:** name = character 3-grams + words, address = tokens, 2^19 buckets per field,
     sublinear tf. For each batch of 2,048 records one sparse × dense product scores all S1 of the country; we keep
     the top 3 by name+address, top 2 by name, top 2 by address.
  2. **n-gram bi-encoder (trained from scratch):** EmbeddingBag over the same n-grams → MLP → 256-d vector, trained with
     InfoNCE on true pairs (in-batch negatives from the same country and state). Five cross-fitted models, so every
     training record is embedded by a model that never saw its entity. Top 5.
  3. The candidate set is the **union** of these lists.
- **Candidate pairs generated:** 83,108,284 on the test set (1,732,544 S1; about 8.3 candidates per record, 48 per
  S1).
- **How we ensured true matches were not lost:** recall is measured on held-out folds: **98.8 %** of true pairs are in
  the candidate set (TF-IDF alone reaches 97.2 % at top 1, 98.6 % at top 5). A perfect scorer on these candidates would
  reach F0.5 0.9963, so the candidates leave little on the table. Most of the remaining misses are records with an
  empty address whose name is shared by many S1 (for example a very common shop name), which no text similarity can
  place.

---

## 4. Matching Model

**Features used (~120):**
- **Name features:** rapidfuzz ratio / token-sort / token-set / partial / Jaro-Winkler on the core name, on the name
  without generic words and without spaces; token Jaccard and containment both ways; first-token match; tokens without
  a fuzzy partner on each side; legal form present / same / added / dropped; domain-name and native-script flags; name
  specificity (how many S1 share the name key).
- **Address features:** token-set / sort / partial ratios, Jaccard and containment, house number equal / edit distance /
  relative difference, numbers compared token by token (`C-9-7` = `C-97`) with overlap and conflicts, state agreement,
  empty address, street core.
- **Other:** TF-IDF and bi-encoder cosines, retrieval ranks in each list; **competition features** (gap to the record's
  best and runner-up S1, the S1's best record, number of candidates on each side); record ambiguity from the retrieval
  lists; two cross-encoder scores; **collective features** (does the S1's other likely records share this record's
  exact address, house number or name key, same or other source; how many confident records the S1 already has per
  source).

**Model type:**
1. **Stage 1:** XGBoost (CUDA, depth 9, early stopping), with and without the embedding features, blended 0.75 / 0.25
   (0.5 / 0.5 for a country without training labels, i.e. France).
2. **Cross-encoders** on the uncertain pairs (stage-1 p between 0.01 and 0.99): `xlm-roberta-base` (MIT) and
   `bge-reranker-v2-m3` (Apache-2.0), fine-tuned on raw text ("name | address" of both sides). On these hard pairs the
   bge-reranker reaches AUC 0.967 against 0.944 for stage 1.
3. **Two collective rounds:** XGBoost on stage-1 score + context + sibling + collective features + both cross-encoder
   scores; round 2 recomputes the collective evidence from round-1 probabilities.

Every model is **cross-fitted** (half A trains on folds 1-2, half B on folds 3-4, each scores the other folds), so all
features seen in training are out-of-fold, like on test. The scores we report (folds 0 and 4) come from models that
never saw those entities, and neither fold is used for early stopping or calibration.

**Threshold selection method:** isotonic calibration (fitted on folds 1-2 of the density-matched universe), each record
goes to its best S1, per-source caps (≤ 5 S2, ≤ 6 S3 per S1), then **expected-F0.5 decoding**: for each S1 we choose
the number of its best records that maximises E[F0.5] under the calibrated probabilities (Monte Carlo on the GPU). The
decoder's settings and the whole V6 plan (caps on, prior correction off, the same setting for France) were chosen by
rules written down before the results existed.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9914** on the density-matched validation (fold 0: 0.99143, fold 4: 0.99127).
  Pair precision 0.9985, pair recall 0.977. US 0.9916, India 0.9913.

| system, scored on the same density-matched validation | F0.5 (min of folds 0 / 4) |
|---|---|
| V2 chain (as it runs on test) | 0.98680 |
| V3 without GNN (as it runs on test) | 0.98981 |
| **V6** (round 2, caps) | **0.99127** |
| V6 with round 1 only | 0.99128 (fold 0: 0.99148) |
| perfect scorer on the candidates (ceiling) | 0.99635 |

- **Common false positives (wrong merges):** records that belong to no S1 but copy an existing S1's name and address
  almost exactly (true duplicates in the source data), and chain names at the same street with a changed house number.
  Only ~1,400 false positives on a whole validation fold.
- **Common false negatives (missed matches):** recall is where the remaining loss is (~22 k FN vs ~1.4 k FP per
  fold).
  (1) Records with an empty address and a very common name (on average 82 S1 share such a name key), which even a
  perfect model cannot place; (2) a changed house number together with a changed or abbreviated name, where the
  evidence stays ambiguous; (3) a completely different trade name at the same address.

---

## 6. Conclusion

Understanding the data mattered more than bigger models: learned transliteration, number-aware address features and,
above all, a validation set that reproduces the test's distractor share *and* its density. Once training matched the
test's crowding, the gap between validation and test largely closed, and pretrained multilingual cross-encoders plus
sibling evidence gave the final gains. The main lessons: validate on data that looks like the test, cross-fit every
learned feature, and give precision-heavy metrics a calibrated decoder rather than a hand-picked threshold.

---

## Appendix

### A. Code Artefacts

Everything is in `code/business_entity_resolution/` (all source in `src/`, with `README.md` and pinned
`requirements.txt`). One command rebuilds both output files from the challenge data:

```bash
cd code/business_entity_resolution
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt
bash run_all.sh --data /path/to/dataset          # resumable; ~30 h on one H100 MIG slice
```

The final files are written to `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Main modules:
`src/common` (normalisation, features, metric), `src/blocking` (GPU TF-IDF), `src/embeddings` (bi-encoders),
`src/xgboost` (pair models), `src/neural_reranker` (cross-encoders), `src/v6` (density-matched universe, collective
rounds, decoding, final file). Models: `xlm-roberta-base` (MIT, 278 M) and `bge-reranker-v2-m3` (Apache-2.0, 568 M),
both fine-tuned on our training pairs; everything else is trained from scratch. No external data or services.

### B. Additional Results

| blocking recall (validation) | top 1 | top 5 | top 20 |
|---|---|---|---|
| TF-IDF name + address | 0.972 | 0.986 | 0.991 |
| n-gram bi-encoder | 0.968 | 0.984 | 0.991 |

| component (hard pairs, validation) | AUC |
|---|---|
| stage-1 XGBoost | 0.944 |
| bge-reranker-v2-m3 (fine-tuned) | 0.967 |

Final test output: 5,836,570 matches; 99,777 S1 (5.8 %) predicted without a match; 723 records removed by the
per-source caps.
