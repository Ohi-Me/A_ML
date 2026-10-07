import os

files = [
    "data/cache_v2/norm_v2_train_s1.parquet",
    "data/cache_v2/norm_v2_train_s2.parquet",
    "data/cache_v2/norm_v2_train_s3.parquet",
    "data/cache_v2/norm_v2_test_s1.parquet",
    "data/cache_v2/norm_v2_test_s2.parquet",
    "data/cache_v2/norm_v2_test_s3.parquet",
    "data/cache_v2/blocking/b1_train/tfidf_name.npz",
    "data/cache_v2/blocking/b1_train/tfidf_addr.npz",
    "data/cache_v2/blocking/b1_train/rid_name_idx.npy",
    "data/cache_v2/blocking/b1_train/rid_name_sc.npy",
    "data/cache_v2/blocking/b1_train/rid_addr_idx.npy",
    "data/cache_v2/blocking/b1_train/rid_addr_sc.npy",
    "data/cache_v2/blocking/b1_test/tfidf_name.npz",
    "data/cache_v2/blocking/b1_test/tfidf_addr.npz",
    "data/cache_v2/blocking/b1_test/rid_name_idx.npy",
    "data/cache_v2/blocking/b1_test/rid_name_sc.npy",
    "data/cache_v2/blocking/b1_test/rid_addr_idx.npy",
    "data/cache_v2/blocking/b1_test/rid_addr_sc.npy",
    "data/cache_v2/emb/e2f_train/emb.npy",
    "data/cache_v2/emb/e2f_test/emb.npy",
    "data/cache_v2/drop_19.npy",
    "data/cache_v2/feats/c2_train_sim19/part_000.parquet",
    "data/cache_v2/feats/c2_test/part_000.parquet",
    "data/cache_v2/feats/oof_xgb_c2_blend_s2_v2.parquet",
    "data/cache/train_gt_pairs.parquet",
    "results/xgboost/xgb_c2_sim19_v2/report.json",
    "results/xgboost/xgb_c2_sim19_noemb_v2/report.json",
    "results/xgboost/xgb_c2_blend_s2_v2/decode_eval.json",
    "data/dataset/test/test_source1.tsv"
]

missing = [f for f in files if not os.path.exists(f)]
if missing:
    print("MISSING:", missing)
else:
    print("all reusable inputs present")

train_parts = len(os.listdir("data/cache_v2/feats/c2_train_sim19")) if os.path.exists("data/cache_v2/feats/c2_train_sim19") else 0
test_parts = len(os.listdir("data/cache_v2/feats/c2_test")) if os.path.exists("data/cache_v2/feats/c2_test") else 0
print(f"c2_train_sim19 parts: {train_parts} (expected 43)")
print(f"c2_test parts: {test_parts} (expected 42)")
