# Experiment log — Business Entity Resolution (Amazon ML Challenge 2026)

Target: macro F0.5 over Source-1 entities (singletons included). Leaderboard top to beat: **0.9768**.
All compute runs as PBS GPU jobs on the NITJ H100 cluster (MIG 3g.47gb slice, 46 GiB), driver `run_h100.py`.

## Validation protocol
- Entity-level folds: `fold = crc32(S1 id) % 5`; fold 0 = validation (~441k S1), folds 1-4 = fitting.
  An S1 and all its true matches share a fold. Unmatched S2/S3 records have no fold.
- Search space during validation = **all** training S1 and **all** training S2/S3 records (same shape as the test
  problem: 1.7M S1 vs 10M records), so competition from other entities and distractors is realistic.
- Scored with the official macro F0.5 (`code/common/metrics.py`), plus pair precision/recall, FP, FN,
  predicted matches, predicted-empty entities, singleton accuracy, candidate recall.
- Anything learned (transliteration maps, models, thresholds) is fitted on folds 1-4 only. IDF weights are
  label-free corpus statistics computed on the corpus being searched (train corpus for validation, test corpus
  for test), which mirrors the test setting and is needed for the unseen French vocabulary.
- Country shift check (France is unseen): train-on-one-country / score-on-the-other diagnostics.

## Data facts (EDA, `results/eda/`)
| | S1 | S2 | S3 |
|---|---|---|---|
| train | 2,206,821 (US 1.32M, India 0.88M) | 5,034,616 | 5,285,603 |
| test | 1,732,544 (India 0.81M, US 0.66M, **France 0.26M**) | 4,887,273 | 5,082,316 |

- 7,638,365 true pairs; singletons 5.6%; matches per S1 mostly 2-6 (max 11); per source up to 5 (S2) / 6 (S3).
- Every S2/S3 record matches **at most one** S1; 26% of S2/S3 records are unmatched distractors.
- True pairs always share the country label -> country blocking is lossless (open set: any string).
- Only ~11% of true pairs have the same lower-cased name; ~4-10% the same address.
- ~10% of S2/S3 names are in Indic scripts (Devanagari, Telugu, Kannada, Tamil, Gujarati, Bengali, ...), mostly
  word-by-word transliterations of the S1 name; S1 names are always Latin.
- Name noise: honorific / symbol prefixes (Mr, Dr, Smt, Shri, M/s, `--`, `>>`, `***`), legal suffix swap /
  drop / duplication, word shuffles, OCR digit swaps (Y0uth, 6lobal, lnc), domain forms (name.com), appended IDs
  and phones, generic word swaps (Academy->Center), DBA / completely different names at the same address.
- Address noise: component reordering, abbreviations (Rd/Road, Ave/Avenue), state as code / full / native
  script, dropped house numbers, added PMB / Unit / Fl, leading zeros (04300), typos, empty addresses (~3%).

## Experiments
| id | date | preproc | blocking | features / model | cand. recall | P | R | F0.5 (val) | FP | FN | runtime | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0-R1..R4 | 09-25 | v1 | exact keys (polars joins) | deterministic rules, 1 S1 per record, ambiguous keys skipped | – | R3: 0.973 | R3: 0.434 | **0.641** (R3), 0.594 (R4), 0.340 (R2) | R3 18.6k | R3 865k | 29 s | exact keys are precise but miss >50% of pairs -> fuzzy retrieval + learned scoring needed (`results/baseline/`) |
| E1 | 09-25 | v1 | learned n-gram bi-encoder (EmbeddingBag on hashed tf-idf n-grams -> MLP -> 256d), InfoNCE, state-bucketed in-batch negatives, 6 epochs, 71 s training; fp16 matmul top-k per country | – | rec@1 0.967, @5 0.983, @10 0.987, @20 0.990 (val) | | | | | train 71 s + retrieval 460 s | train folds rec@1 0.980 -> some overfit; `results/embeddings/e1/` |
| E2 | 09-25 | v1 | bi-encoder as E1, 20 epochs + 15% n-gram dropout (3 min training) | – | rec@1 0.9685, @5 0.9842, @10 0.9877, @20 0.9908 (val) | | | | | train 3 min + retrieval 7 min | small gain over E1; chosen for candidates. Miss analysis @5 (24.2k of 1.53M val pairs): 68% have an EMPTY record address with a name shared by several S1 (inherently ambiguous -> abstain), rest DBA names with partial address. `results/embeddings/e2/` |
| B1 | 09-25 | v1 | hashed TF-IDF (name char-3g + words, address tokens), cuSPARSE SpMM + fp16 contiguous top-k per country, lists name/addr/comb top-20 per record | val rec@1/@5/@20: comb 0.9715/0.9856/0.9908, name 0.502/0.682/0.779, addr 0.790/0.882/0.908 | | | | | | ~50 min on a 3g.40gb slice | top-k along the strided dim was 5x slower than on a contiguous fp16 copy (bench_spmm.py) |
| C1 | 09-25 | v1 | union: e2 emb top-5 + b1 comb top-3 + name top-2 + addr top-2 | – | **val cand. recall 0.9881** (all folds 0.9921) | | | | | | 26 s | 84.8M pairs, 8.2 per record, 39.7 per S1 (p99 238, max 7962); 95.9% of val S1 have every match covered |
| X1 (xgb_c1) | 09-25 | v1 | C1 | 69 features (tf-idf + e2 cosines, retrieval ranks, record/S1 competition gaps & ranks, rapidfuzz name/addr, token sets, legal, numbers, state, name frequency); XGBoost CUDA depth 9, eta 0.06, early stop -> 1798 rounds; 2-half cross-fit; decision: record -> best S1 if p>=0.70 and margin over runner-up >=0.20 (tuned on folds 1-2) | 0.9881 | 0.9962 | 0.9588 | **0.98296** | 5,630 | 62,898 | features 8 min, XGB 11 min | tune folds 0.9870 vs val 0.9830: in-sample e2 cosines on training folds (see E3). Singleton acc 0.982. FP: 75% are records of no S1 that look like a real S1 (confident p~1; mining shows 0.17% error among 7.3M confident pairs, spread thinly -> mostly irreducible duplicates); FN: 18.2k blocking (80% empty address), 32k below threshold (house-number typos), 12.5k lost to another S1. Top gain: b_gap2_cos_e2, b_rank_cos_e2, num_conflict, a_empty2 |
| X1-S2 (xgb_c1_s2) | 09-25 | v1 | C1 | stage 2 on X1 OOF p: record competition in probability space (best / 2nd p, margin, rank), S1 context (sum p, #confident, #claimed, other max), sibling similarity (emb / tf-idf name / addr to the S1's other confident records); XGB 726 rounds, same cross-fit | 0.9881 | 0.9971 | 0.9661 | **0.98506** (thr .70/m .20) | 4,251 | 51,883 | 30 min (siblings on GPU) | gain almost all from s1_p + s2_b_margin; siblings add little. 1-match S1 bucket drops to ~0.93 while singletons rise to ~0.99 |
| D1 | 09-25 | – | – | decision layer on X1-S2: (a) threshold+margin grid widened -> thr .65 / margin .50: 0.98528; (b) **expected-F0.5 decoding** per S1 (records to best S1, isotonic calibration on folds 1-2, choose top-k maximizing E[F0.5] by GPU Monte Carlo, gamma 1.0) | | 0.9975 | 0.9646 | **0.98544** | 3,717 | 54,146 | 8 min | principled, no hand threshold; favors precision |
| SHIFT | 09-25 | v1 | – | covariate / prior shift checks (`results/eda/shift_check_c1.json`, `addr_format.json`) | | | | | | | | address/name formats identical train vs test for US/India; France: no state detected (state_cmp=0) and smaller best-vs-runner-up margins. **Test has 5.76 records per S1 vs 4.68 in train while predicted matches per S1 stay ~3.35 -> ~40% distractor records in test vs 26% in train.** |
| SIM19 | 09-25 | v1 | C1 minus dropped S1 | simulate test prior: drop 19% of S1 (hash), their records become distractors; ranks + competition features recomputed (`simulate_drop.py`) | 0.9881 | 0.9943 | 0.9593 | X1 models on SIM19 val: **0.98171** (vs 0.98296 on normal val; thr .70 / margin .60 tuned under SIM19) | 6,777 | 50,489 | 1 min | 40.5% distractors after the drop, as in test. Use SIM19 for model selection, calibration and final training. |
| E3 | 09-25 | v1 | 5 cross-fitted bi-encoders (each record queried with the model that never saw its entity; pair cosine from the S1-fold model; test: all-folds e2f for retrieval, mean of fold models for the cosine) | – | OOF rec@1 train folds 0.9685 = val 0.9686 (E2: 0.981 vs 0.969) | | | | | | 18 min | in-sample gap removed |
| C2 | 09-25 | v1 | e3 OOF emb top-5 + b1 comb 3 / name 2 / addr 2 | – | val 0.98815, all folds 0.98808 | | | | | | | 84.8M pairs |
| X1-SIM19 | 09-25 | v1 | C1 minus dropped | X1 features, trained under SIM19 | 0.9881 | 0.9958 | 0.9565 | 0.98180 (SIM19 val) | 4,982 | 53,883 | 15 min | = model trained on full data and tuned under SIM19 (0.98171): what matters is tuning / calibrating under the test prior |
| **X2-SIM19** (xgb_c2_sim19) | 09-25 | v1 | C2 (OOF e3) | F2 features (+ code-number overlap/conflict, house-number edit distance / relative diff, legal-family relation, token alignment) + OOF e3 cosine; 2153 rounds | 0.9882 | 0.9962 | 0.9644 | **0.98519** (SIM19 val; +0.34 pts over X1-SIM19) | 4,566 | 44,162 | feats 23 min, XGB 20 min | F2 features = 33% of gain (cnum_conflict #2 overall); top: b_gap2_cos_sum 26%, cnum_conflict 16%, cos_e3 9% |

### Operational notes (25 Sep)
- 10:45:01 and 12:44:02: every running job of ours was SIGKILLed in the same second (PBS mom log: external kill, jobs far under their 32 GB). Cause not visible to us; jobs now wait if their expected run would cross the next xx:44 window of an even hour, and every stage writes results incrementally.
- Worker pools hang on this node (named semaphores in /dev/shm disappear): feature and prep stages now run in one process (polars / rapidfuzz threads).
- Two jobs once landed on the same MIG slice (race); slices are now claimed with lock files.
- 13:26 the shared root disk hit 100% (our cache 101 GB). Removed superseded c1-era artifacts and v1 intermediates (kept only what the v1-c2 fallback needs); stage 2 now writes side-car columns only. Never touched other folders (WidarH100 belongs to another project).
- Normalization v2 (French regions/departments -> state, whole-part state names): test S1 with no state 259,253 -> 0.
| D2 | 09-25 | v1 | C2 | decision rules on X2-SIM19 (stage 1 only): threshold+margin (thr .70 / m .40) val 0.98519 vs expected-F decoding 0.98483 -> without stage 2 the explicit margin wins; the final predictor now picks the rule with the better tuning-fold score | | | | | | | | |
| SUB-2 (final_c2v1) | 09-25 | v1 | C2 test (83.1M pairs) | X2-SIM19 refit on all folds (x1.25 rounds), expected-F decoding (job started before the rule patch) | | | | sim-val 0.9848 | | | 16 min | validator PASS; 5.80M matches, 96k empty S1, France 94.5% matched. `submissions/final_c2v1/` |
| X2-NOEMB | 09-25 | v1 | C2 | X2-SIM19 without any embedding feature (cos_e3 + its gaps/ranks + r_e3_emb) | 0.9882 | 0.9959 | 0.9630 | 0.98443 (vs 0.98519 with embeddings) | 4,918 | 45,873 | 20 min | embedding worth ~0.08 pts on US/India |
| AGREE | 09-25 | v1 | C2 test | decisions of X2-SIM19 vs X2-NOEMB on the test candidates, per country (label-free) | | | | | | | | disagreement US 0.58%, India 1.19%, **France 3.23%**; on US/India the embedding model mostly adds accepts, in France the differences are symmetric -> embedding possibly uninformative on French text (France also has smaller margins) |
| BLEND | 09-25 | v1 | C2 | stage-1 blend w*full + (1-w)*no-embedding, thr/margin tuned on folds 1-2 (SIM19) | | | | w=1: 0.98519, **w=0.75: 0.98530** (best tune+val), w=0.5: 0.98515, w=0: 0.98443 | 5,074 (w .75) | 42,543 | 2 min | blending is free on US/India -> final: w=0.75 for countries with training labels, w=0.5 for countries without (France), stage 2 without embedding features so the hedge is kept |
| V2 chain | 09-25 | v2 | C2 (v2) | full rerun with normalization v2: blocking -> e3/e2f -> candidates -> F2 features -> SIM19 -> full + no-emb stage 1 -> blend -> stage 2 (no emb features) -> decoding -> final | | | | (pending) | | | ~3.5 h wall | v2 train blocking comb rec@1 0.97166 / @10 0.98847 (v1 0.97150 / 0.98838) |
| X2-D11 | 09-25 | v1 | C2 | X2-SIM19 with depth 11, eta 0.05 | | | | 0.98513 (vs 0.98519 depth 9) | 5,230 | 42,542 | 10 min | no gain -> keep depth 9 |
| HEADROOM | 09-25 | v1 | C2 | p of each val record's best candidate: true vs wrong | | | | | | | | confident errors small (p>0.95: ~1.2k wrong), uncertain band 0.2-0.95 holds ~42k true / ~26k wrong -> a better pair model has room |
| XENC | 09-25 | v1 | C2 uncertain pairs (0.02<=p<=0.98: 911k, 38% true) | char-level cross-encoder from scratch (4 layers, d 256, bf16), 2 epochs per half on ~366k pairs, cross-fitted; logistic stack with stage-1 logit | | | | stage-1 0.98519 -> **0.98531** with xenc (tune 0.98527 -> 0.98534) | 4,408 | 44,109 | 7 min train | weak alone (AUC 0.66 vs 0.91 on these pairs) but complementary; not in the final (extra test stages for ~+0.01) |
| SHIFT-v2 | 09-25 | v2 | C2 v2 | top-candidate feature distributions by country (`results/eda/shift_check_c2_test_v2.json`) | | | | | | | | France state_cmp mean 0.001 (v1) -> 0.639 (v2); France top-candidate a_tset 88.3 -> 93.4, cos_addr 0.776 -> 0.821; test embedding margins now match train for US/India (OOF fix) |
| X2-v2 / NOEMB-v2 / BLEND-v2 | 09-25 | v2 | C2 v2 (84.9M pairs) | stage 1 full (1958 rounds) / no-embedding / 0.75 blend, SIM19 | 0.9883 | 0.9956 (blend) | 0.9662 | full **0.98521**, no-emb 0.98447, blend **0.98529** | 5,268 | 41,868 | 20 min each | v2 = v1 on US/India (as intended) |
| **S2-v2** (xgb_c2_blend_s2_v2) | 09-25 | v2 | C2 v2 | stage 2 on the blended stage-1 p, no embedding features (92 features, siblings from e2f) | 0.9883 | 0.9968 | 0.9696 | **0.98691** (SIM19 val) | 3,915 | 37,615 | 25 min | singleton acc 0.990 |
| **GNN-v2** | 09-25 | v2 | C2 v2 graph (68.5M edges, 2.2M S1, 10.3M records) | local edge GNN (your architecture): edge = blended stage-1 logit + 24 key features; 2 rounds of mean/max pooling over S1 and record nodes; S1 mini-batches with 2-hop neighbourhoods on the GPU; cross-fitted halves; 4 epochs, ~5 min per half | 0.9883 | | | **0.98829** (thr .70 / m .50; vs XGB stage 2 0.98691) | 2,954 | 34,303 | 12 min | val AP 0.99967 vs 0.99933 for its input; giant-degree S1 (>5000 candidates) scored alone |
| GNN-v2 + decode | 09-25 | v2 | C2 v2 | GNN scores -> isotonic (folds 1-2) -> expected-F0.5 decoding (gamma 1) | 0.9883 | 0.9978 | 0.9713 | **0.98841** (threshold rule 0.98831) | 2,673 | 35,518 | 8 min | best single system so far |
| MLXENC (xlm-roberta-base) | 09-25 | v2 | C2 v2 uncertain pairs (0.02<=p<=0.98: 981k) | pretrained multilingual cross-encoder (MIT, 278M) fine-tuned on raw text (native scripts, accents), 2 epochs, 400k pairs per half, bf16, cross-fitted | | | | on uncertain val pairs: AUC 0.945 vs 0.916 stage 1, AP 0.922 vs 0.882 | | | 33 min | far better than the from-scratch char model (AUC 0.66); feeds the fusion layer |
| FUSE-LR | 09-25 | v2 | C2 v2 | logistic fusion of GNN prob + XLM-R cross-encoder (uncertain pairs), fitted on folds 1-2; threshold rule chosen on tuning | 0.9883 | 0.9983 | 0.9755 | **0.99064** | 2,103 | 30,338 | 5 min | +0.23 over GNN alone |
| **GNN-MLX** (final v3) | 09-25 | v2 | C2 v2 | GNN with the cross-encoder logit (+flag) as extra edge input -> isotonic -> expected-F0.5 decoding (tuning 0.99075 > threshold 0.99071) | 0.9883 | 0.9987 | 0.9744 | **0.99078** | 1,629 | 31,722 | 10 min + 8 | best system; val AP 0.99978 |
| SUB-3 (final_v2) | 09-25 | v2 | C2 v2 test (83.1M pairs) | stage 1 full + no-emb refit on all folds (x1.25 rounds), blended 0.75 seen / 0.5 unseen countries; stage 2 (no emb); isotonic + expected-F decoding | | | | system val 0.98696 | | | 50 min | validator PASS; 5.84M matches, 98.8k empty S1. `submissions/final_v2/` |
| **SUB-4 (final_v3) — FINAL** | 09-25 | v2 | C2 v2 test | test stage-1 blend (from final_v2) -> XLM-R cross-encoder on 1.29M uncertain test pairs -> GNN-MLX (mean of 2 halves) -> isotonic -> expected-F0.5 | 0.9883 | 0.9987 | 0.9744 | **system val 0.99078** | 1,629 | 31,722 | ~1 h | validator PASS; 5.91M matches, 99.5k empty S1 (5.7%); France 94.1% matched. `submissions/final_v3/`, package `submissions/team_submission.zip` |

## Summary (25 Sep, end of day)
Validated progression (test-prior simulation, fold 0): exact keys 0.641 -> stage 1 C1 0.9818 -> C2 (OOF embeddings,
number/legal/token features) 0.9852 -> stage 2 0.9869 -> GNN 0.9884 -> GNN + multilingual cross-encoder 0.9908.
Submissions: final_c1 (0.983 plain val), final_c2v1 (0.9848 sim), final_v2 (0.9870 sim), **final_v3 (0.9908 sim)**.

## Phase 2 (25 Sep, evening): V3 leaderboard regression (0.99078 val → 0.982814 LB, V2 0.98696 → 0.98332)
| id | what | result |
|---|---|---|
| SUBDIFF | label-free diff of the submitted files (`code/audit/submission_diff.py`) | V3 vs V2: +106,409 / −39,770 pairs (net +66,639, +1.14 %); 99.4 % of V3-only accepts are records V2 left unassigned, 98.2 % land on S1 V2 already matched; empty rate 5.705 % → 5.745 %. Validation predicted only +5,379 on 358 k S1 (≈ +26 k at test scale) → test change ≈ 2.5× validation |
| CODE-AUDIT | train/test pipeline mismatches (see `PHASE2.md` §2) | M1 test `cos_e3` = mean of 5 fold models vs single held-out model in validation (GNN edge input in V3; dropped by V2's stage 2); M2 refit stage 1 vs half models; M3 XLM-R mean of halves + band from refit; M4 GNN mean of halves vs single chain on validation; M5 r_e3_emb from e2f on test |
| PHASE2-SUITE | `code/phase2/` + `run_all.sh` + `H100_MANUAL.md`: cosine variants, 10-way ablation, extra-accept trace, India→US LOCO, chain-A V4 candidates (V4A/B/C), pre-registered selection | written and CPU smoke-tested on synthetic data (`code/phase2/tests/smoke.sh`); **labelled runs pending on the GPU** |
| LB-V4A | V4A (chain-A replay + validation-style cosine) submitted | leaderboard **0.981** (V3 0.982814, V2 0.98332): more test accepts -> lower LB; the chain/cosine hypothesis is not the cause |
| COS-RESULT | `results/phase2/cos/cos_report.json` (H100 run) | 5-model mean vs validation-style cosine on test: Pearson 0.96 (all pairs) / 0.985 (top pair), mean diff 0.035 -> minor |
| V5 | `V5.md`, `code/v5/`: V2 recipe + 21 transferable features (ambiguity, specificity, street core), LOCO-chosen unseen-country decoder, France pseudo-labels gated by India->US LOCO, pre-registered choice | smoke-tested on CPU; **H100 run pending** (`bash run_all.sh --stage v5all`) |
