#!/bin/bash
# Submit the whole pipeline for one normalization version as a chain of PBS GPU jobs (run from the laptop).
#   bash code/final/submit_chain.sh v2 [after_jobid]
# Every stage writes to data/cache_<v> (derived artifacts) and results/<experiment>/<tag>_<v>.
set -e
V=$1
AFTER=$2
ENV="--env ER_NORM=$V --env ER_CACHE=data/cache_$V"
S="py -3.10 run_h100.py submit"
dep() { if [ -n "$1" ]; then echo "--after $1"; fi; }
sub() { local name=$1; shift; local after=$1; shift; $S ${name}_$V $ENV $(dep $after) "$@" | tail -1; }

# blocking (tf-idf lists) on both splits
B1=$(sub blk_tr "$AFTER" --hours 3 --ncpus 32 --mem 32gb --expect 60 -- python -u code/blocking/sparse_tfidf.py --split train --tag b1 --no_s1side --k_rid 20)
B2=$(sub blk_te "$AFTER" --hours 3 --ncpus 32 --mem 32gb --expect 45 -- python -u code/blocking/sparse_tfidf.py --split test --tag b1 --no_s1side --k_rid 20)
# cross-fitted encoders (train) + all-folds encoder for test retrieval
E3=$(sub e3 "$B1" --hours 3 --ncpus 8 --mem 32gb --expect 25 -- python -u code/embeddings/oof_biencoder.py --split train --tag e3)
EF=$(sub e2f "$B1" --hours 1 --ncpus 8 --mem 32gb --expect 6 -- python -u code/embeddings/ngram_biencoder.py --mode train --tag e2f --epochs 20 --drop 0.15 --final --train_only)
EFR=$(sub e2f_tr "$EF" --hours 1 --ncpus 8 --mem 32gb --expect 10 -- python -u code/embeddings/ngram_biencoder.py --mode embed --split train --tag e2f --model_from e2f)
EFT=$(sub e2f_te "$EF:$B2" --hours 1 --ncpus 8 --mem 32gb --expect 8 -- python -u code/embeddings/ngram_biencoder.py --mode embed --split test --tag e2f --model_from e2f)
# candidates
C1=$(sub cands_tr "$E3" --hours 1 --ncpus 8 --mem 32gb --expect 5 -- python -u code/blocking/build_candidates.py --name c2 --split train --lists emb/e3:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2)
C2=$(sub cands_te "$EFT" --hours 1 --ncpus 8 --mem 32gb --expect 5 -- python -u code/blocking/build_candidates.py --name c2 --split test --lists emb/e2f:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2)
# OOF pair cosines
P1=$(sub pc_tr "$C1" --hours 1 --ncpus 8 --mem 32gb --expect 12 -- python -u code/embeddings/oof_biencoder.py --split train --tag e3 --reuse --cands c2)
P2=$(sub pc_te "$C2:$E3" --hours 1 --ncpus 8 --mem 32gb --expect 5 -- python -u code/embeddings/oof_biencoder.py --split test --tag e3 --cands c2)
# features
F1=$(sub feats_tr "$P1" --hours 3 --ncpus 32 --mem 32gb --expect 30 -- python -u code/tfidf_fuzzy/pair_features.py --cands c2 --split train --btag b1 --etags "" --fset v2 --pairc emb/e3_train/paircos_c2.parquet)
F2=$(sub feats_te "$P2" --hours 3 --ncpus 32 --mem 32gb --expect 30 -- python -u code/tfidf_fuzzy/pair_features.py --cands c2 --split test --btag b1 --etags "" --fset v2 --pairc emb/e3_test/paircos_c2.parquet)
# test-like simulation, stage 1, stage 2, decoding
SM=$(sub sim "$F1" --hours 1 --ncpus 16 --mem 32gb --expect 5 -- python -u code/xgboost/simulate_drop.py --feats c2_train --drop 0.19)
X1=$(sub xgb "$SM" --hours 3 --ncpus 16 --mem 32gb --expect 25 -- python -u code/xgboost/train_xgb.py --feats c2_train_sim19 --tag xgb_c2_sim19_$V --simdrop 19)
X2=$(sub s2 "$X1:$EFR" --hours 3 --ncpus 16 --mem 32gb --expect 35 -- python -u code/xgboost/stage2.py --feats c2_train_sim19 --s1tag xgb_c2_sim19_$V --tag xgb_c2_sim19_s2_$V --etag e2f --simdrop 19)
D2=$(sub dec "$X2" --hours 1 --ncpus 16 --mem 32gb --expect 12 -- python -u code/xgboost/decode_eval.py --tag xgb_c2_sim19_s2_$V --simdrop 19)
echo "chain $V: blk $B1 $B2 | e3 $E3 e2f $EF $EFT | cands $C1 $C2 | pc $P1 $P2 | feats $F1 $F2 | sim $SM | xgb $X1 | s2 $X2 | dec $D2"
