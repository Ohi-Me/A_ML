# Phase 2 — why V3 regressed on the leaderboard, and how V4 is chosen

| system | validation (SIM19, fold 0) | leaderboard |
|---|---|---|
| V2 (stage-1 blend → XGB stage 2 → isotonic → expected-F0.5) | 0.98696 | **0.98332** |
| V3 (stage-1 blend → XLM-R → GNN → isotonic → expected-F0.5) | 0.99078 | 0.982814 |

Status: **the diagnosis code is complete and smoke-tested. The labelled experiments still have to run on the GPU**
(`bash run_all.sh`, see `H100_MANUAL.md`). The run needs an H100 and the 100+ GB cache, which the session that
wrote this code did not have. Sections 1–2 use only files already in this repository. Everything in sections 3–5 is
produced by the runner, and V4 is chosen by the fixed rule in section 5, not by hand.

## 1. What is already established (label-free, from the submitted files and the logs)

`python code/audit/submission_diff.py` → `results/audit/submission_diff.json`

* V3 accepts **106,409** pairs that V2 does not, and drops 39,770 that V2 accepted: net **+66,639** matches (+1.14 %).
* **99.4 %** of the V3-only accepts are records that V2 left unassigned. Only 591 are records moved between S1.
* **98.2 %** of them land on S1 entities that V2 had already matched. Only 1,904 go to S1 that V2 left empty. The
  predicted-empty (singleton) rate barely moves (5.705 % → 5.745 %). The k-histogram shifts right: entities
  with 5, 6, 7 and 8 predicted matches grow by 2.5 %, 4.8 %, 7.1 % and 10.3 %.
  → V3's regression isn't about singletons. **V3 adds a 4th/5th/6th record to entities that are already matched.**
  Under F0.5 each such false positive costs about 0.17 on that entity.
* Validation predicted a much smaller change. V3 made 1,209,069 predictions vs 1,203,690 for V2 on 357,966 fold-0 S1
  (+5,379, +0.45 %), and false positives *fell* from 3,660 to 1,629. Scaled to 1.73 M test S1 that is ≈ +26 k. The
  observed +66.6 k is **≈ 2.5× what validation predicted**. The US share (+49 k on 0.66 M S1) comes from the
  25 Sep gap audit (US uncertain share of assigned records 3.2 % on validation → 7.8 % on test; India normal).

## 2. Train/test pipeline mismatches found in the code

For each score a downstream model consumes, this compares how it was made on validation and on test:

| # | input | validation / training | test (V2, V3) | affects |
|---|---|---|---|---|
| M1 | bi-encoder cosine `cos_e3` and its competition features (`b_gap`, `b_gap2`, `a_gap`, ranks) | the **one** fold model that never saw the S1's fold (`oof_biencoder.py`: `a_fold == f`). Competing S1s in other folds use *different* models | **mean of the 5 fold models' cosines** (`--split test`): a lower-noise, differently distributed feature, and the competition compares averaged cosines | stage 1 (V2 and V3); **GNN raw edge input (V3 only; V2's stage 2 drops every embedding feature)** |
| M2 | stage-1 probability | half models A/B (40 % of S1 each), out of fold | **refit on all folds with 1.25× the rounds**: sharper scores | V2 stage 2 and V3 GNN (edge input 0), XLM-R band selection, calibration |
| M3 | XLM-R cross-encoder logit | one half model; band chosen from the OOF stage-1 p | **mean of both halves' logits**; band chosen from the refit p | V3 only |
| M4 | GNN logit | fold 0 scored by half A only; isotonic fitted on single-model scores | **mean of halves A and B** | V3 only |
| M5 | retrieval rank `r_e3_emb` | rank in the record's e3 OOF top-k | e2f all-folds model's rank, renamed | stage-1 full model (V2 and V3) |

M1, M3 and M4 all average several models' scores on test while the downstream model and the calibration only ever
saw single-model scores. Averaging lowers noise, so the best competing candidate's score drops and margins look
cleaner. That's consistent with V3 accepting more borderline extra records on already-matched entities.
M1 (GNN input), M3 and M4 are V3-only, which fits "V2 generalised, V3 didn't".
These are hypotheses until the ablations below quantify them.

## 3. Experiments the runner performs (same C2 candidate population throughout)

**Bi-encoder discrepancy** (`code/phase2/cos_variants.py`). Recomputes all five fold-model cosines for every test
pair and checks that their mean equals the `cos_e3` that V2/V3 used (the smoke test verifies the code path
exactly). It then builds the validation-style variant `hash1` (model `fold_of(S1 id)`, the same hash as the
training folds) with the same competition features, and reports:
distribution shift (quantiles, KS vs validation fold 0, per country, all pairs and each record's top pair),
Pearson/Spearman between the two computations, the between-model spread, how often a record's top-cosine S1
changes, and re-scored stage 1 (refit and half A) with `hash1`.

**Ablation matrix** (`code/phase2/ablation.py` → `results/phase2/ablation.json/.csv`). For each system it reports
validation F0.5 on the clean folds 0 and 4 (fold 3 flagged: it was XGBoost's early-stopping fold) per country, a
bootstrap CI, test matches, empty rate, per-country change vs V2, how many of the 106 k V3-only accepts disappear,
the test/validation inflation of the change vs V2, and the uncertain-share ratio.

| request | system name | note |
|---|---|---|
| 1 V2 baseline | `V2` | replay checked against `submissions/final_v2` |
| 2 V3 full | `V3` | replay checked against `submissions/final_v3` |
| 3 V3 without GNN | `V3_noGNN` | logistic stack of stage-1 + XLM-R |
| 4 V3 without XLM-R | `V3_noXLMR` | = gnn_v2 |
| 5 GNN without XLM-R edge score | `V3_noXLMR` | the same model as 4: in V3, XLM-R only enters as a GNN edge input |
| 6 GNN without bi-encoder cosine | `V3_GNNnoCos` | retrained, cross-fitted exactly like V3 |
| 7 GNN with validation-style cosine | `V3_cosHash1_GNN` (GNN edges only), `V3_cosHash1_chain` (stage 1 + XLM-R band + GNN) | no retraining |
| 8 V3 with V2 decoder | `V3_V2decoder` | V2 and V3 use the same decoder settings (isotonic folds 1–2, expected-F0.5 γ=1), so this row must equal V3: it checks that the decoder is not the cause |
| 9 V3 without isotonic | `V3_noIso` | |
| 10 V3 threshold + margin | `V3_thrMargin` | tuned on folds 1–2 |
| chain consistency (M2) | `V3_stage1halfA/B` | 25 Sep audit jobs gnn_cfA / gnn_cfB (not re-run) |
| V4 candidates | `V4A`, `V4B`, `V4C` | section 5 |

**Pre-registered interpretation.** If a cosine variant (row 7) removes **≥ 25 % of the US V3-only accepts** while
leaving validation unchanged, M1 is a primary bug rather than a distribution shift. The same threshold applies to
M2 (`V3_stage1halfA`) and to the combined chain replay (`V4A`).

**Extra-accept trace** (`code/phase2/diff_accepts.py` → `results/phase2/diff_accepts.json`). Every pair in V3-only,
V2-only, common, rejected, and a population sample, on test and on validation fold 0 (with labels there), traced
through: stage-1 p (refit and half A), candidate ranks, cosine (both computations), XLM-R, GNN logit, calibrated p,
V2 calibrated p, and expected-F0.5 position/k. For each variable it computes KS(validation group, test group) minus
KS(validation population, test population). The largest group-specific shift is the stage where V3-only accepts
diverge.

**Leave-one-country-out** (`code/phase2/loco_eval.py`). Every learned stage is fitted on India only and evaluated
on US fold 0 (unseen) next to India fold 0 (in-domain), for: stage 1, stage 1 + XLM-R stack, GNN without XLM-R,
GNN + XLM-R. Each is decoded with isotonic + EF, raw + EF, and threshold + margin, and the run also reports AP/AUC,
calibration error and predicted/true match ratio. There are two runs: a full run, and `--noemb`, which avoids
leakage from the bi-encoder trained on both countries. The key output is `component_deltas_iso_ef`: does adding
the GNN or XLM-R help the *unseen* country, or only the in-domain one?

## 4. Anti-overfitting protocol

* Two clean held-out folds (0 and 4) plus a bootstrap CI, instead of one split. The V4 score is the **minimum**
  over folds 0 and 4.
* Label-free test/validation consistency is a hard gate: this is exactly the symptom that exposed V3.
* LOCO is a hard gate for each learned component.
* The selection rule was written before any phase-2 number existed and is kept in code (`select_v4.py`).

## 5. V4 — candidates and the fixed selection rule

The principle, from section 2: **on test, replay exactly the chain that produced the validation scores.** Chain A
(the models that scored folds 0 and 4) is stage-1 half A → XLM-R half A on A's own band → GNN half A, with the
validation-style cosine. There's no refit and no averaging of halves, so the calibration map and decoder settings
fitted on validation apply to test scores of the same kind.

| option | V4 candidate | implementation |
|---|---|---|
| A corrected V3 | `V4A` | V3 checkpoints, chain A replay |
| B simplified V3 | `V4B` | GNN without XLM-R, chain A replay |
| C stronger V2-style | `V4C` | V2 stage-1 half A → stage-2 model A, chain A replay |
| D/E new retrieval / stronger cross-encoder | deferred | only if the ablations show the architecture, not the pipeline, limits generalisation (see below) |

Rule (`code/phase2/select_v4.py`):
1. **R = min(F0.5 fold 0, F0.5 fold 4).**
2. **Consistency gate:** for US and India, the test/validation ratio of predicted matches per S1 may differ from
   V2's by at most 0.004, and the uncertain share on test may be at most 1.5× its validation value.
3. **LOCO gate:** the GNN (V4A, V4B) and XLM-R (V4A) may not lower unseen-country F0.5 by more than 0.001.
4. The highest R wins, but it must beat V2 (and any simpler candidate) by more than 0.0003. Otherwise the simpler
   system wins. If nothing passes, **V2 stays**.

The final command is written to `results/phase2/v4_choice.json`. `run_all.sh` executes it and validates the file.

### Stronger models (after the diagnosis)

These candidates satisfy the licence rule (MIT/Apache-2.0, ≤ 8 B): XLM-R-large (MIT, 560 M),
BAAI/bge-reranker-v2-m3 (Apache-2.0, 568 M), intfloat/multilingual-e5-large (MIT), Qwen3-Reranker-0.6B
(Apache-2.0). `ml_cross_encoder.py` takes the model name from one constant, so a larger reranker is a small change.
None will be added unless it beats the chain-A V4 on folds 0 **and** 4 and passes the LOCO gate. The 25 Sep
headroom analysis (≈ 42 k true / 26 k wrong pairs in the 0.2–0.95 band) shows room for a better pair model, but the
leaderboard gap measured so far comes from the test pipeline, not from model capacity.
**Nothing here supports a claim that ~0.995 is reachable yet.**
