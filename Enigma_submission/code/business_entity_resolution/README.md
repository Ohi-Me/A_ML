# Business Entity Resolution — Team Enigma (ML Challenge 2026)

This folder is the complete, runnable pipeline behind our final submission (the **V6** system). It rebuilds both
submission files, `output/matching_results.tsv` and `output/candidate_pairs.tsv`, from the challenge's training and
test files alone: no external data, no external APIs. Validation macro F0.5 of V6: **0.9914** (test-density
validation, see `Documentation_template.md` §5).

## 1. What you need

| | minimum | what we used |
|---|---|---|
| GPU | one NVIDIA GPU with >= 40 GB, CUDA 12.x driver | H100 NVL (MIG 3g.47gb slice, 46 GB) |
| CPU / RAM | 8 cores, 64 GB RAM (32 cores help the feature steps) | 8 cores, 32 GB per job |
| disk | ~250 GB free for the cache (`data/cache_v2`) | |
| internet | once, to download the two pretrained cross-encoders from Hugging Face | |
| OS / Python | Linux, Python 3.10, bash | Ubuntu, Python 3.10.21 |

## 2. Set up

```bash
cd code/business_entity_resolution
python3.10 -m venv .venv && source .venv/bin/activate          # or: conda create -n er python=3.10
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Put the challenge files anywhere, laid out as provided:
```
<dataset>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<dataset>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```
For the built-in format check, copy the official validator to `data/utils/validate_submission.py`
(or keep the challenge's `Data/student_resource/` folder next to this README; the runner links it).

## 3. Run

```bash
bash run_all.sh --data <dataset>            # full pipeline, resumable
bash run_all.sh --list                      # every step with its exact command
nohup bash run_all.sh --data <dataset> > run.out 2>&1 &     # detached; follow with: tail -f run.out
```

When it finishes, the two submission files are in `output/`:
`output/matching_results.tsv` and `output/candidate_pairs.tsv` (copied from `results/final/final_v6/output/`).

* **Resumable:** each finished step leaves a marker in `data/cache_v2/_done/`. If the machine or job stops, run the
  same command again; finished steps are skipped. To redo one step: `bash run_all.sh --only <step>`.
* **Logs:** `results/_runs/<step>.log`. Reports of every model: `results/<experiment>/`.
* **Run time** on one H100 MIG slice: about 30 h for everything (blocking and pair features ~6 h, cross-encoders
  ~3 h, the V6 blocks ~12 h; the rest are the earlier versions whose outputs V6 reuses). A full H100 is faster.
* **Randomness is fixed:** folds are a hash of the entity id and model seeds are fixed, so a rerun gives the same
  files up to GPU floating-point order.

## 4. What runs, in order

| block | what it does | main code |
|---|---|---|
| `v2` | normalisation (learned transliteration of Indic scripts, legal forms, address words, states / French regions), GPU TF-IDF retrieval, n-gram bi-encoders, candidate union, pair features, stage-1 XGBoost | `src/common`, `src/blocking`, `src/embeddings`, `src/tfidf_fuzzy`, `src/xgboost` |
| `v3` | multilingual cross-encoder `xlm-roberta-base` on the uncertain pairs | `src/neural_reranker` |
| `audit`, `phase2`, `loco`, `select`, `v4` | diagnostics of the validation / test gap, India -> US unseen-country check | `src/audit`, `src/phase2` |
| `v6diag` | data probe; the **density-matched training universe** (test density per country) | `src/v6/simulate_universe.py`, `score_chain.py` |
| `v6feat` | transferable features (record ambiguity, name specificity, street core) | `src/v5/extra_feats.py` |
| `v6s1` | stage 1: XGBoost with and without embedding features, blended | `src/xgboost/train_xgb.py`, `blend_oof.py` |
| `v6x` | second cross-encoder `bge-reranker-v2-m3` fine-tuned on the uncertain band | `src/neural_reranker/ml_cross_encoder.py` |
| `v6col` | two collective rounds (sibling evidence) + decoder search | `src/v6/train_collective.py`, `collective.py` |
| `v6loco` | India -> US check that decides the unseen-country (France) settings | `src/phase2/loco_eval.py` |
| `v6test` | the same chain on the test set | `src/v6/predict_v6.py` |
| `v6final`, `v6pick` | plan fixed in advance, final file, official validator, copy to `output/` | `src/v6/choose_v6.py`, `final_v6.py` |

## 5. Models and licences

| model | size | licence | use |
|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | 278 M | MIT | cross-encoder, fine-tuned on our training pairs |
| `BAAI/bge-reranker-v2-m3` | 568 M | Apache-2.0 | second cross-encoder, fine-tuned on our training pairs |
| n-gram bi-encoders, XGBoost models | small | ours | trained from scratch on the challenge data |

Both pretrained models are well under 8B parameters. All Python packages are BSD / MIT / Apache-2.0 / ISC.

## 6. Folder layout

```
code/business_entity_resolution/
  run_all.sh          one runner for the whole pipeline
  requirements.txt    pinned packages
  src/                all source code (the runner exposes it as code/, which is how the modules import each other)
    common/           io, normalisation, transliteration, folds, pair features, metric, decision rules
    blocking/         GPU TF-IDF retrieval, candidate union
    embeddings/       n-gram bi-encoders (cross-fitted)
    tfidf_fuzzy/      pair feature builder
    xgboost/          stage-1 / stage-2 models, drop simulation, decoding evaluation, error analysis
    neural_reranker/  multilingual cross-encoders
    gnn/              edge GNN on the candidate graph (V3; not used by V6)
    audit/, phase2/   diagnostics of the validation / test gap, unseen-country checks
    v5/, v6/          transferable features, density-matched universe, collective rounds, V6 decoding and choice
    final/            test prediction, submission writer, validator wrapper
```
