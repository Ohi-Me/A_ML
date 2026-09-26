#!/usr/bin/env bash
# =====================================================================================================================
#  Business Entity Resolution - one self-contained runner for a single GPU machine (H100 or any CUDA GPU >= 40 GB).
#  No SSH, no scheduler, no hard-coded paths: run it from the repository folder (or anywhere, it cds to itself).
#
#  bash run_all.sh --data /path/to/dataset [--stage all|v6all|<block>] [--only STEP] [--dry]
#
#    --data DIR   folder that contains train/ and test/ with the challenge TSVs (train_source1.tsv ... test_source3.tsv,
#                 train_ground_truth.tsv). Linked to data/dataset (the code reads data/dataset). Needed only once.
#    --stage      which block to run. all (default) = v2 -> v3 -> audit -> phase2 -> loco -> select -> v4 -> V6 blocks.
#                 V6 on an existing V2/V3/phase-2 cache: --stage v6all (v6diag -> v6feat -> v6s1 -> v6x -> v6col ->
#                 v6loco -> v6test -> v6final -> v6pl -> v6pick, see V6.md). Optional blocks, never part of all/v6all:
#                 V5 (--stage v5all) and v6a (--stage v6a, the V2 recipe at test density: a fallback / ablation)
#    --only STEP  run a single step by name (see --list), even if it was done before
#    --list       print every step with its block and exit
#    --dry        print what would run
#  Environment (optional): ER_CACHE (default data/cache_v2), NCPUS (default: all cores), HF_HOME (default ./hf_cache)
#
#  Every step writes a marker data/cache_v2/_done/<step> when it succeeds and is skipped next time (also skipped when its
#  main output already exists, so an existing cache from the cluster is reused). Logs: results/_runs/<step>.log.
#  If a step was killed half way, just run the same command again. To redo a step: --only <step>.
#  See H100_MANUAL.md for set-up, run times, outputs and what to send back.
# =====================================================================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA=""; STAGE="all"; ONLY=""; DRY=0; LIST=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --dry) DRY=1; shift;;
    --list) LIST=1; shift;;
    -h|--help) sed -n 2,22p "$0"; exit 0;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done

export ER_ROOT="$ROOT" ER_NORM=v2 ER_CACHE="${ER_CACHE:-data/cache_v2}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
NCPUS="${NCPUS:-$(nproc)}"
export NCPUS OMP_NUM_THREADS="$NCPUS" POLARS_MAX_THREADS="$NCPUS" RAYON_NUM_THREADS="$NCPUS"
C="$ER_CACHE"
mkdir -p data results/_runs "$C/_done"

# ---------------------------------------------------------------- data + validator links
if [[ -n "$DATA" ]]; then
  DATA="$(cd "$DATA" && pwd)"
  ln -sfn "$DATA" data/dataset
fi
[[ -e data/utils ]] || ln -sfn "$ROOT/Data/student_resource/utils" data/utils

# ---------------------------------------------------------------- step helper
declare -a ORDER=()
step() {   # step BLOCK NAME CHECK_PATH "command"
  local block="$1" name="$2" check="$3" cmd="$4"
  ORDER+=("$block $name")
  if [[ $LIST == 1 ]]; then printf "%-8s %-22s %s\n" "$block" "$name" "$cmd"; return 0; fi
  if [[ -n "$ONLY" ]]; then [[ "$ONLY" == "$name" ]] || return 0
  else
    [[ ( "$STAGE" == "all" && ! "$block" =~ ^(v5|loco5|v5pl|v5final|v6a|v7diag)$ ) || "$STAGE" == "$block" ||
       ( "$STAGE" == "v5all" && "$block" =~ ^(v5|loco5|v5pl|v5final)$ ) ||
       ( "$STAGE" == "v6all" && "$block" =~ ^(v6diag|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ ) ]] || return 0
    if [[ -e "$C/_done/$name" || ( -n "$check" && -e "$check" ) ]]; then echo "[skip] $name"; return 0; fi
  fi
  echo "[run ] $(date '+%H:%M:%S') $name"
  [[ $DRY == 1 ]] && { echo "       $cmd"; return 0; }
  local t0=$SECONDS
  { echo "== $(date -Iseconds) $name"; echo "$cmd"; } >> "results/_runs/$name.log"
  bash -c "$cmd" >> "results/_runs/$name.log" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $name (exit $rc) after $((SECONDS - t0)) s - last lines of results/_runs/$name.log:"
    tail -n 25 "results/_runs/$name.log"
    exit $rc
  fi
  touch "$C/_done/$name"
  echo "[ ok ] $name in $((SECONDS - t0)) s"
}

preflight() {
  [[ $LIST == 1 || $DRY == 1 ]] && return 0
  # raw TSVs are needed unless their parquet copies (data/cache/*.parquet, made on first read) already exist
  if [[ ! -f data/cache/train_source1.parquet ]]; then
    for f in train/train_source1.tsv train/train_source2.tsv train/train_source3.tsv train/train_ground_truth.tsv \
             test/test_source1.tsv test/test_source2.tsv test/test_source3.tsv; do
      [[ -f "data/dataset/$f" ]] || { echo "missing data/dataset/$f - pass --data /path/to/dataset (folder with train/ and test/)"; exit 3; }
    done
  fi
  python - <<'EOF' || exit 4
import torch, xgboost, polars, sklearn, scipy, rapidfuzz, anyascii, transformers  # noqa
assert torch.cuda.is_available(), "CUDA is not available: this pipeline needs an NVIDIA GPU"
p = torch.cuda.get_device_properties(0)
print(f"GPU {p.name} {p.total_memory / 2**30:.0f} GiB | torch {torch.__version__} | xgboost {xgboost.__version__}")
EOF
}
preflight

P=python\ -u
DROP_EMB="cos_e3,b_gap_cos_e3,b_rank_cos_e3,a_gap_cos_e3,a_rank_cos_e3,b_gap2_cos_e3,r_e3_emb"
XTR="xenc/ml_hardv2_train_scores.parquet:mlxenc"; XTE="xenc/ml_hardv2_test_scores.parquet:mlxenc"

# ================================================ V2 (known-good baseline, rebuilt from the raw TSVs) =================
step v2 prep_train      "$C/norm_v2_train_s3.parquet" "$P code/common/prep.py --split train --dict folds"
step v2 prep_test       "$C/norm_v2_test_s3.parquet"  "$P code/common/prep.py --split test --dict all"
step v2 blk_train       "$C/blocking/b1_train/rid_comb_idx.npy" "$P code/blocking/sparse_tfidf.py --split train --tag b1 --no_s1side --k_rid 20"
step v2 blk_test        "$C/blocking/b1_test/rid_comb_idx.npy"  "$P code/blocking/sparse_tfidf.py --split test --tag b1 --no_s1side --k_rid 20"
step v2 e3_train        "$C/emb/models/e3_fold4.pt" "$P code/embeddings/oof_biencoder.py --split train --tag e3"
step v2 e2f_train       "$C/emb/models/e2f.pt" "$P code/embeddings/ngram_biencoder.py --mode train --tag e2f --epochs 20 --drop 0.15 --final --train_only"
step v2 e2f_emb_train   "$C/emb/e2f_train/emb.npy" "$P code/embeddings/ngram_biencoder.py --mode embed --split train --tag e2f --model_from e2f"
step v2 e2f_emb_test    "$C/emb/e2f_test/emb.npy"  "$P code/embeddings/ngram_biencoder.py --mode embed --split test --tag e2f --model_from e2f"
step v2 cands_train     "$C/cands/c2_train.parquet" "$P code/blocking/build_candidates.py --name c2 --split train --lists emb/e3:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2"
step v2 cands_test      "$C/cands/c2_test.parquet"  "$P code/blocking/build_candidates.py --name c2 --split test --lists emb/e2f:emb:5,blocking/b1:comb:3,blocking/b1:name:2,blocking/b1:addr:2"
step v2 paircos_train   "$C/emb/e3_train/paircos_c2.parquet" "$P code/embeddings/oof_biencoder.py --split train --tag e3 --reuse --cands c2"
step v2 paircos_test    "$C/emb/e3_test/paircos_c2.parquet"  "$P code/embeddings/oof_biencoder.py --split test --tag e3 --cands c2"
step v2 feats_train     "" "$P code/tfidf_fuzzy/pair_features.py --cands c2 --split train --btag b1 --etags '' --fset v2 --pairc emb/e3_train/paircos_c2.parquet"
step v2 feats_test      "" "$P code/tfidf_fuzzy/pair_features.py --cands c2 --split test --btag b1 --etags '' --fset v2 --pairc emb/e3_test/paircos_c2.parquet"
step v2 sim19           "" "$P code/xgboost/simulate_drop.py --feats c2_train --drop 0.19"
step v2 xgb_full        "$C/feats/oof_xgb_c2_sim19_v2.parquet" "$P code/xgboost/train_xgb.py --feats c2_train_sim19 --tag xgb_c2_sim19_v2 --simdrop 19"
step v2 xgb_noemb       "$C/feats/oof_xgb_c2_sim19_noemb_v2.parquet" "$P code/xgboost/train_xgb.py --feats c2_train_sim19 --tag xgb_c2_sim19_noemb_v2 --simdrop 19 --drop $DROP_EMB"
step v2 blend           "$C/feats/oof_xgb_c2_blend_v2.parquet" "$P code/xgboost/blend_oof.py --full xgb_c2_sim19_v2 --noemb xgb_c2_sim19_noemb_v2 --w 0.75 --out xgb_c2_blend_v2"
step v2 stage2          "$C/feats/oof_xgb_c2_blend_s2_v2.parquet" "$P code/xgboost/stage2.py --feats c2_train_sim19 --s1tag xgb_c2_blend_v2 --tag xgb_c2_blend_s2_v2 --etag e2f --simdrop 19 --drop_feats $DROP_EMB,sib_emb_max,sib_emb_mean"
step v2 dec_v2          "" "$P code/xgboost/decode_eval.py --tag xgb_c2_blend_s2_v2 --simdrop 19"
step v2 final_v2        "$C/feats/test_scores_final_v2.parquet" "$P code/final/predict_test_v2.py --s1_feats c2_train_sim19 --s1_tag xgb_c2_sim19_v2 --blend_from xgb_c2_blend_v2 --s2_tag xgb_c2_blend_s2_v2 --test_feats c2_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v2"
step v2 val_v2          "" "bash code/final/validate.sh results/final/final_v2/output"

# ================================================ V3 (GNN + XLM-R; reproduces the 0.982814 submission) ===============
step v3 gnn_v2          "$C/gnn/gnn_v2_train_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_train_sim19 --s1 oof_xgb_c2_blend_v2.parquet --tag gnn_v2 --batch_s1 1500"
step v3 gnn_v2_oof      "$C/feats/oof_gnn_v2.parquet" "$P code/gnn/make_oof.py oof_xgb_c2_blend_v2.parquet gnn/gnn_v2_train_scores.parquet gnn_v2"
step v3 gnn_v2_dec      "" "$P code/xgboost/decode_eval.py --tag gnn_v2 --sub gnn --simdrop 19"
step v3 mlx_prep        "$C/xenc/hardv2_train.parquet" "$P code/neural_reranker/prep_pairs.py --split train --scores oof_xgb_c2_blend_v2.parquet --name hardv2 --raw --maxlen 110"
step v3 mlx_train       "$C/xenc/ml_hardv2_train_scores.parquet" "$P code/neural_reranker/ml_cross_encoder.py --name hardv2 --split train --epochs 2"
step v3 gnn_mlx         "$C/gnn/gnn_mlx_v2_train_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_train_sim19 --s1 oof_xgb_c2_blend_v2.parquet --tag gnn_mlx_v2 --batch_s1 1500 --extra $XTR"
step v3 gnn_mlx_oof     "$C/feats/oof_gnn_mlx_v2.parquet" "$P code/gnn/make_oof.py oof_xgb_c2_blend_v2.parquet gnn/gnn_mlx_v2_train_scores.parquet gnn_mlx_v2"
step v3 gnn_mlx_dec     "" "$P code/xgboost/decode_eval.py --tag gnn_mlx_v2 --sub gnn --simdrop 19"
step v3 mlx_prep_te     "$C/xenc/hardv2_test.parquet" "$P code/neural_reranker/prep_pairs.py --split test --scores test_stage1_final_v2.parquet --name hardv2 --raw --maxlen 110"
step v3 mlx_score_te    "$C/xenc/ml_hardv2_test_scores.parquet" "$P code/neural_reranker/ml_cross_encoder.py --name hardv2 --split test --score_only"
step v3 gnn_v2_te       "$C/gnn/gnn_v2_test_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_final_v2.parquet --tag gnn_v2 --split test --batch_s1 1500"
step v3 gnn_mlx_te      "$C/gnn/gnn_mlx_v2_test_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_final_v2.parquet --tag gnn_mlx_v2 --split test --batch_s1 1500 --extra $XTE"
step v3 final_v3        "" "$P code/final/final_from_scores.py --test_scores gnn/gnn_mlx_v2_test_scores.parquet --col gnn --logit --oof oof_gnn_mlx_v2.parquet --decode results/gnn/gnn_mlx_v2/decode_eval.json --out final_v3"
step v3 val_v3          "" "bash code/final/validate.sh results/final/final_v3/output"

# ================================================ audit (jobs from 25 Sep, kept as they were) ========================
step audit gap_audit    "$C/feats/test_stage1_halfB_v2.parquet" "$P code/audit/gap_audit.py"
step audit gnn_cfA      "$C/gnn/gnn_cfA_test_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_halfA_v2.parquet --tag gnn_mlx_v2 --split test --batch_s1 1500 --extra $XTE --halves A --out_tag gnn_cfA && $P code/audit/test_decisions.py --scores gnn/gnn_cfA_test_scores.parquet --col gnn --logit --oof oof_gnn_mlx_v2.parquet --name cfA"
step audit gnn_cfB      "$C/gnn/gnn_cfB_test_scores.parquet" "$P code/gnn/edge_gnn.py --feats c2_test --s1 test_stage1_halfB_v2.parquet --tag gnn_mlx_v2 --split test --batch_s1 1500 --extra $XTE --halves B --out_tag gnn_cfB && $P code/audit/test_decisions.py --scores gnn/gnn_cfB_test_scores.parquet --col gnn --logit --oof oof_gnn_mlx_v2.parquet --name cfB"

# ================================================ phase 2: bi-encoder cosine, ablations, V4 candidates ===============
step phase2 cos5        "$C/emb/e3_test/paircos5_c2.parquet" "$P code/phase2/cos_variants.py cos"
step phase2 cosvar      "$C/p2/cosvar/c2_test_hash1.parquet" "$P code/phase2/cos_variants.py variants --vars mean5,hash1"
step phase2 cos_report  "" "$P code/phase2/cos_variants.py report"
step phase2 s1_hash1    "" "$P code/phase2/cos_variants.py stage1 --var hash1 --models refit,halfA"
step phase2 s1_mean5chk "" "$P code/phase2/cos_variants.py stage1 --var mean5 --models refit"
step phase2 gnn_nocos   "$C/feats/oof_p2_gnn_mlx_nocos.parquet" "$P code/phase2/gnn_run.py train --tag gnn_mlx_nocos --extra $XTR --drop_feats cos_e3"
step phase2 gnn_nocos_te "$C/p2/gnn/gnn_mlx_nocos_test_scores.parquet" "$P code/phase2/gnn_run.py score --ckpt p2/gnn/gnn_mlx_nocos --split test --s1 test_stage1_final_v2.parquet --extra $XTE --out gnn_mlx_nocos"
step phase2 v3_gnnhash1 "$C/p2/gnn/v3_gnnhash1_test_scores.parquet" "$P code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --split test --s1 test_stage1_final_v2.parquet --extra $XTE --override hash1 --out v3_gnnhash1"
step phase2 v4_prep     "" "$P code/phase2/v4.py prep"
step phase2 xenc_chainA "$C/p2/xenc_chainA_hash1.parquet" "$P code/neural_reranker/xenc_score_pairs.py --split test --pairs p2/v4_pairs_halfA_hash1.parquet --jobs xA=A@pA --out p2/xenc_chainA_hash1.parquet"
step phase2 xenc_refit  "$C/p2/xenc_refit_hash1.parquet" "$P code/neural_reranker/xenc_score_pairs.py --split test --pairs p2/v4_pairs_refit_hash1.parquet --jobs xA=A@pR,xB=B@pR --out p2/xenc_refit_hash1.parquet"
step phase2 xmean       "" "$P code/phase2/v4.py xmean"
step phase2 v3_chainhash1 "$C/p2/gnn/v3_chainhash1_test_scores.parquet" "$P code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --split test --s1 p2_test_s1_refit_hash1.parquet --extra p2/xenc_refit_hash1_mlx.parquet:mlxenc --override hash1 --out v3_chainhash1"
step phase2 v4a         "$C/p2/gnn/v4a_chainA_hash1_test_scores.parquet" "$P code/phase2/gnn_run.py score --ckpt gnn/gnn_mlx_v2 --halves A --split test --s1 p2_test_s1_halfA_hash1.parquet --extra p2/xenc_chainA_hash1_mlx.parquet:mlxenc --override hash1 --out v4a_chainA_hash1"
step phase2 v4b         "$C/p2/gnn/v4b_gnnA_hash1_test_scores.parquet" "$P code/phase2/gnn_run.py score --ckpt gnn/gnn_v2 --halves A --split test --s1 p2_test_s1_halfA_hash1.parquet --override hash1 --out v4b_gnnA_hash1"
step phase2 v4c         "$C/feats/p2_test_v4c_hash1_scores.parquet" "$P code/phase2/v4.py stage2 --var hash1"
step phase2 ablation    "" "$P code/phase2/ablation.py"
step phase2 diff_accepts "" "$P code/phase2/diff_accepts.py"

# ================================================ leave-one-country-out (India -> US) ================================
step loco loco_noemb    "" "$P code/phase2/loco_eval.py all --src India --dst US --noemb"
step loco loco_full     "" "$P code/phase2/loco_eval.py all --src India --dst US"

# ================================================ V4: pre-registered choice, then the submission files ===============
step select select_v4   "" "$P code/phase2/select_v4.py"
V4CMD='python -u - <<EOF
import json, subprocess, sys
c = json.load(open("results/phase2/v4_choice.json"))
print("V4 choice:", c["choice"]); cmd = c["final_command"]
if not cmd.startswith("python"):
    print("V2 stays the best robust system; no V4 file written"); sys.exit(0)
sys.exit(subprocess.call(cmd, shell=True))
EOF'
step v4 final_v4        "" "$V4CMD"
step v4 val_v4          "" "[ -f results/final/final_v4/output/matching_results.tsv ] || exit 0; bash code/final/validate.sh results/final/final_v4/output && mkdir -p submissions/final_v4 && { [ -e submissions/final_v4/matching_results.tsv ] || cp results/final/final_v4/output/matching_results.tsv submissions/final_v4/; }"

# ================================================ V5: V2 recipe + transferable features, LOCO-checked unseen-country
#                                                  handling, optional France pseudo-labels (see V5.md)
V5DROP="$DROP_EMB,sib_emb_max,sib_emb_mean"
step v5 v5_feats_train  "" "$P code/v5/extra_feats.py --src c2_train_sim19"
step v5 v5_feats_test   "" "$P code/v5/extra_feats.py --src c2_test"
step v5 v5_xgb_full     "$C/feats/oof_xgb_v5_full.parquet" "$P code/xgboost/train_xgb.py --feats c2v5_train_sim19 --tag xgb_v5_full --simdrop 19"
step v5 v5_xgb_noemb    "$C/feats/oof_xgb_v5_noemb.parquet" "$P code/xgboost/train_xgb.py --feats c2v5_train_sim19 --tag xgb_v5_noemb --simdrop 19 --drop $DROP_EMB"
step v5 v5_blend        "$C/feats/oof_xgb_v5_blend.parquet" "$P code/xgboost/blend_oof.py --full xgb_v5_full --noemb xgb_v5_noemb --w 0.75 --out xgb_v5_blend"
step v5 v5_stage2       "$C/feats/oof_xgb_v5_s2.parquet" "$P code/xgboost/stage2.py --feats c2v5_train_sim19 --s1tag xgb_v5_blend --tag xgb_v5_s2 --etag e2f --simdrop 19 --drop_feats $V5DROP"
step v5 v5_dec          "" "$P code/xgboost/decode_eval.py --tag xgb_v5_s2 --simdrop 19"
step v5 v5_recall       "" "$P code/v5/recall_report.py --s1 xgb_v5_blend --s2 xgb_v5_s2"
step v5 v5_test         "$C/feats/test_scores_final_v5.parquet" "$P code/final/predict_test_v2.py --s1_feats c2v5_train_sim19 --s1_tag xgb_v5_full --blend_from xgb_v5_blend --s2_tag xgb_v5_s2 --test_feats c2v5_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v5"
step loco5 loco_v2feat  "" "$P code/phase2/loco_eval.py all --skip_neural --name v2"
step loco5 loco_v5feat  "" "$P code/phase2/loco_eval.py all --skip_neural --name v5 --feats_table c2v5_train_sim19 --full_tag xgb_v5_full --noemb_tag xgb_v5_noemb"
step v5pl pl_select     "" "$P code/v5/pseudo_label.py select"
step v5pl pl_fit        "" "$P code/v5/pseudo_label.py fit"
step v5pl pl_score      "$C/feats/test_scores_final_v5pl.parquet" "$P code/v5/pseudo_label.py score"
step v5final final_v5   "" "$P code/v5/final_v5.py --scores feats/test_scores_final_v5.parquet --loco_system stage1_all --out final_v5"
step v5final final_v5pl "" "$P code/v5/final_v5.py --scores feats/test_scores_final_v5pl.parquet --loco_system stage1_pl --out final_v5pl"
step v5final val_v5     "" "bash code/final/validate.sh results/final/final_v5/output && bash code/final/validate.sh results/final/final_v5pl/output"
step v5final choose_v5  "" "$P code/v5/choose_v5.py"

# ================================================ V6 diagnostics (see V6_RESEARCH.md): cheap, decisive
step v6diag v6_probe     "" "$P code/v6/data_probe.py"
step v6diag v6_v3nognn   "" "$P code/v6/submit_from_decisions.py --decisions p2/decisions/V3_noGNN_test.parquet --out final_v3nognn && bash code/final/validate.sh results/final/final_v3nognn/output"
step v6diag v6_dms       "" "$P code/v6/simulate_universe.py --src c2_train --plan US:0.62:0.19,India:1.0:0.19 --tag dmsA --code 62"
step v6diag v6_dms_chain "" "$P code/v6/score_chain.py --feats c2_train_dmsA --code 62 --name dmsA"
# ================================================ V6a: the V2 recipe trained and calibrated in the test-density universe
step v6a v6a_full   "$C/feats/oof_xgb_v6a_full.parquet" "$P code/xgboost/train_xgb.py --feats c2_train_dmsA --tag xgb_v6a_full --simdrop 62"
step v6a v6a_noemb  "$C/feats/oof_xgb_v6a_noemb.parquet" "$P code/xgboost/train_xgb.py --feats c2_train_dmsA --tag xgb_v6a_noemb --simdrop 62 --drop $DROP_EMB"
step v6a v6a_blend  "$C/feats/oof_xgb_v6a_blend.parquet" "$P code/xgboost/blend_oof.py --full xgb_v6a_full --noemb xgb_v6a_noemb --w 0.75 --out xgb_v6a_blend --simdrop 62"
step v6a v6a_stage2 "$C/feats/oof_xgb_v6a_s2.parquet" "$P code/xgboost/stage2.py --feats c2_train_dmsA --s1tag xgb_v6a_blend --tag xgb_v6a_s2 --etag e2f --simdrop 62 --drop_feats $DROP_EMB,sib_emb_max,sib_emb_mean"
step v6a v6a_dec    "" "$P code/xgboost/decode_eval.py --tag xgb_v6a_s2 --simdrop 62"
step v6a v6a_test   "$C/feats/test_scores_final_v6a.parquet" "$P code/final/predict_test_v2.py --s1_feats c2_train_dmsA --s1_tag xgb_v6a_full --blend_from xgb_v6a_blend --s2_tag xgb_v6a_s2 --test_feats c2_test --rename r_e2f_emb:r_e3_emb --etag_test e2f --out final_v6a"
step v6a v6a_val    "" "bash code/final/validate.sh results/final/final_v6a/output"

# ================================================ V6 (V6.md): the chain trained and calibrated at test density
#   universe (v6diag) -> transferable features (v6feat) -> stage 1 (v6s1) -> second cross-encoder (v6x, optional:
#   XENC2_MODEL=none switches it off, a failure falls back to XLM-R base) -> collective sibling rounds 1-2 (v6col) ->
#   India->US LOCO for the unseen country (v6loco) -> test (v6test) -> pre-registered plan + file (v6final) ->
#   France pseudo-labels if the plan allows (v6pl) -> validate + pick (v6pick)
V6DROP="$DROP_EMB,sib_emb_max,sib_emb_mean"
XENC2="${XENC2_MODEL:-BAAI/bge-reranker-v2-m3}"
NOX2="[ -f $C/xenc/ml_v6band_model_A.pt ] || { echo 'no second cross-encoder: skipped'; exit 0; }"
PLON="$P code/v6/choose_v6.py pl_on || { echo 'plan: no pseudo-labels, skipped'; exit 0; }"
FARGS="\$($P code/v6/choose_v6.py final_args)"
step v6feat v6_feats_train "" "$P code/v5/extra_feats.py --src c2_train_dmsA --absent_code 62 --out c2v6_train_dmsA"
step v6feat v6_feats_test  "" "$P code/v5/extra_feats.py --src c2_test --out c2v6_test"
step v6s1 v6_full       "$C/feats/oof_xgb_v6_full.parquet"  "$P code/xgboost/train_xgb.py --feats c2v6_train_dmsA --tag xgb_v6_full --simdrop 62"
step v6s1 v6_noemb      "$C/feats/oof_xgb_v6_noemb.parquet" "$P code/xgboost/train_xgb.py --feats c2v6_train_dmsA --tag xgb_v6_noemb --simdrop 62 --drop $DROP_EMB"
step v6s1 v6_blend      "$C/feats/oof_xgb_v6_blend.parquet" "$P code/xgboost/blend_oof.py --full xgb_v6_full --noemb xgb_v6_noemb --w 0.75 --out xgb_v6_blend --simdrop 62"
step v6x v6x_prep       "" "[ '$XENC2' = none ] && exit 0; $P code/neural_reranker/prep_pairs.py --split train --scores oof_xgb_v6_blend.parquet --name v6band --raw --maxlen 110 --lo 0.01 --hi 0.99"
step v6x v6x_train      "" "[ '$XENC2' = none ] && { echo 'second cross-encoder switched off'; exit 0; }; $P code/neural_reranker/ml_cross_encoder.py --name v6band --split train --epochs 2 --model $XENC2 --col bgexenc || { echo 'WARNING: second cross-encoder failed - V6 continues with XLM-R base only'; rm -f $C/xenc/ml_v6band_train_scores.parquet $C/xenc/ml_v6band_model_A.pt $C/xenc/ml_v6band_model_B.pt; }"
step v6col v6_c1        "$C/feats/oof_xgb_v6_c1.parquet" "$P code/v6/train_collective.py --feats c2v6_train_dmsA --s1tag xgb_v6_blend --tag xgb_v6_c1 --simdrop 62 --drop_feats $V6DROP"
step v6col v6_c1_dec    "" "$P code/xgboost/decode_eval.py --tag xgb_v6_c1 --simdrop 62"
step v6col v6_c2        "$C/feats/oof_xgb_v6_c2.parquet" "$P code/v6/train_collective.py --feats c2v6_train_dmsA --s1tag xgb_v6_blend --prev xgb_v6_c1 --tag xgb_v6_c2 --simdrop 62 --drop_feats $V6DROP"
step v6col v6_c2_dec    "" "$P code/xgboost/decode_eval.py --tag xgb_v6_c2 --simdrop 62"
step v6loco v6_loco     "" "$P code/phase2/loco_eval.py all --skip_neural --name v6 --feats_table c2v6_train_dmsA --full_tag xgb_v6_full --noemb_tag xgb_v6_noemb --simdrop 62"
step v6test v6_test_s1  "$C/feats/test_stage1_final_v6.parquet" "$P code/v6/predict_v6.py --step stage1 --out final_v6"
step v6test v6x_prep_te "" "$NOX2; $P code/neural_reranker/prep_pairs.py --split test --scores test_stage1_final_v6.parquet --name v6band --raw --maxlen 110 --lo 0.01 --hi 0.99"
step v6test v6x_score_te "" "$NOX2; $P code/neural_reranker/ml_cross_encoder.py --name v6band --split test --score_only --model $XENC2 --col bgexenc"
step v6test v6_test_rounds "$C/feats/test_scores_final_v6.parquet" "$P code/v6/predict_v6.py --step rounds --out final_v6"
step v6final v6_plan    "" "$P code/v6/choose_v6.py plan"
step v6final v6_final   "" "$P code/v6/final_v6.py --scores feats/test_scores_final_v6.parquet --tag xgb_v6_c2 $FARGS --out final_v6"
step v6pl v6_pl_select  "" "$PLON; $P code/v6/pl_v6.py select"
step v6pl v6_pl_fit     "" "$PLON; $P code/v6/pl_v6.py fit"
step v6pl v6_pl_score   "" "$PLON; $P code/v6/pl_v6.py score"
step v6pl v6_pl_final   "" "$PLON; $P code/v6/final_v6.py --scores feats/test_scores_final_v6pl.parquet --tag xgb_v6_c2 $FARGS --out final_v6pl"
step v6pick v6_val      "" "for d in final_v6 final_v6pl final_v6a final_v3nognn; do f=results/final/\$d/output; [ -f \$f/matching_results.tsv ] || continue; bash code/final/validate.sh \$f || exit 1; done"
step v6pick v6_choose   "" "$P code/v6/choose_v6.py pick"

# ================================================ V7 diagnostics (cheap, labels of train only): what blocking misses and
#                                                  what each new pass would recover; where V6 loses F0.5; France sample
step v7diag v7_miss_probe "" "$P code/v7/miss_probe.py --universe dms"
step v7diag v7_error_dump "" "$P code/v7/error_dump.py"

if [[ $LIST == 0 && $DRY == 0 ]]; then
  echo "done ($STAGE${ONLY:+, only $ONLY}). V6 choice: results/v6/v6_choice.json (first_choice_file); phase 2: results/phase2/"
fi
