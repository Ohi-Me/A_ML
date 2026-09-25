#!/bin/bash
# End-to-end reproduction without PBS (one NVIDIA GPU, >= 32 GB RAM). Run from the project root, with the
# challenge data under data/dataset/{train,test} and data/utils/validate_submission.py.
#   bash code/final/run_all_local.sh v2
# Every stage is the same script the H100 jobs ran; outputs land in results/final/final_<v>/output/.
set -euo pipefail
V=${1:-v2}
export ER_NORM=$V ER_CACHE=data/cache_$V PYTHONPATH=$(pwd)/code:${PYTHONPATH:-}
py() { echo "== $*"; python -u "$@"; }

# 1. normalization (+ transliteration maps learned from training pairs)
py code/common/prep.py --split train --dict folds
py code/common/prep.py --split test --dict all
# 2. blocking: hashed tf-idf retrieval on the GPU (name, address, name+address lists)
py code/blocking/sparse_tfidf.py --split train --tag b1 --no_s1side --k_rid 20
py code/blocking/sparse_tfidf.py --split test --tag b1 --no_s1side --k_rid 20
# 3. n-gram bi-encoders: 5 cross-fitted models (train) and one all-folds model (test retrieval, siblings)
py code/embeddings/oof_biencoder.py --split train --tag e3
py code/embeddings/ngram_biencoder.py --mode train --tag e2f --epochs 20 --drop 0.15 --final --train_only
py code/embeddings/ngram_biencoder.py --mode embed --split train --tag e2f --model_from e2f
py code/embeddings/ngram_biencoder.py --mode embed --split test --tag e2f --model_from e2f
# 4. candidate pairs (union of lists) and out-of-fold pair cosines
py code/blocking/build_candidates.py --name c2 --split train --lists emb/e3:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2
py code/blocking/build_candidates.py --name c2 --split test --lists emb/e2f:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2
py code/embeddings/oof_biencoder.py --split train --tag e3 --reuse --cands c2
py code/embeddings/oof_biencoder.py --split test --tag e3 --cands c2
# 5. pair features
py code/tfidf_fuzzy/pair_features.py --cands c2 --split train --btag b1 --etags "" --fset v2 --pairc emb/e3_train/paircos_c2.parquet
py code/tfidf_fuzzy/pair_features.py --cands c2 --split test --btag b1 --etags "" --fset v2 --pairc emb/e3_test/paircos_c2.parquet
# 6. test-like training prior (19% of S1 dropped); stage 1 with and without embedding features, their blend,
#    stage 2 (no embedding features), decision settings
EMB=cos_e3,b_gap_cos_e3,b_rank_cos_e3,a_gap_cos_e3,a_rank_cos_e3,b_gap2_cos_e3,r_e3_emb
py code/xgboost/simulate_drop.py --feats c2_train --drop 0.19
py code/xgboost/train_xgb.py --feats c2_train_sim19 --tag xgb_c2_sim19_$V --simdrop 19
py code/xgboost/train_xgb.py --feats c2_train_sim19 --tag xgb_c2_sim19_noemb_$V --simdrop 19 --drop $EMB
py code/xgboost/blend_oof.py --full xgb_c2_sim19_$V --noemb xgb_c2_sim19_noemb_$V --w 0.75 --out xgb_c2_blend_$V
py code/xgboost/stage2.py --feats c2_train_sim19 --s1tag xgb_c2_blend_$V --tag xgb_c2_blend_s2_$V --etag e2f --simdrop 19 --drop_feats $EMB,sib_emb_max,sib_emb_mean
py code/xgboost/decode_eval.py --tag xgb_c2_blend_s2_$V --simdrop 19
# 7. all-folds refit of stage 1 + v2 test prediction (also saves the test stage-1 scores used below)
py code/final/predict_test_v2.py --s1_feats c2_train_sim19 --s1_tag xgb_c2_sim19_$V --blend_from xgb_c2_blend_$V    --s2_tag xgb_c2_blend_s2_$V --test_feats c2_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_$V
# 8. multilingual cross-encoder (xlm-roberta-base, MIT) on the uncertain pairs, cross-fitted
py code/neural_reranker/prep_pairs.py --split train --scores oof_xgb_c2_blend_$V.parquet --name hard$V --raw --maxlen 110
py code/neural_reranker/ml_cross_encoder.py --name hard$V --split train --epochs 2
py code/neural_reranker/prep_pairs.py --split test --scores test_stage1_final_$V.parquet --name hard$V --raw --maxlen 110
py code/neural_reranker/ml_cross_encoder.py --name hard$V --split test --score_only
# 9. local GNN on the candidate graph with the cross-encoder score as an edge input, then decoding settings
py code/gnn/edge_gnn.py --feats c2_train_sim19 --s1 oof_xgb_c2_blend_$V.parquet --tag gnn_mlx_$V --batch_s1 1500    --extra xenc/ml_hard${V}_train_scores.parquet:mlxenc
py code/gnn/make_oof.py oof_xgb_c2_blend_$V.parquet gnn/gnn_mlx_${V}_train_scores.parquet gnn_mlx_$V
py code/xgboost/decode_eval.py --tag gnn_mlx_$V --sub gnn --simdrop 19
py code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_final_$V.parquet --tag gnn_mlx_$V --split test --batch_s1 1500    --extra xenc/ml_hard${V}_test_scores.parquet:mlxenc
# 10. calibration + expected-F0.5 decoding on the test scores -> submission files -> validator
py code/final/final_from_scores.py --test_scores gnn/gnn_mlx_${V}_test_scores.parquet --col gnn --logit    --oof oof_gnn_mlx_$V.parquet --decode results/gnn/gnn_mlx_$V/decode_eval.json --out final_v3
python -u data/utils/validate_submission.py --matching results/final/final_v3/output/matching_results.tsv    --candidate results/final/final_v3/output/candidate_pairs.tsv --test-dir data/dataset/test --check-ids
