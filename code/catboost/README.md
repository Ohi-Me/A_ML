# catboost — not run (decision, 25 Sep 2026)

The pair model is XGBoost on the GPU (code/xgboost). A second gradient-boosted learner on the same features was not
trained because:
- CatBoost GPU training failed to parse the MIG device ids of the cluster (CUDA_VISIBLE_DEVICES=MIG-...);
- a deeper XGBoost (depth 11) gave no gain (0.98513 vs 0.98519), so more tree capacity on the same features is not
  the bottleneck;
- the remaining errors needed different information, which came from models that do add diversity: the local GNN
  on the candidate graph (+0.14 over the XGBoost stage 2) and the multilingual cross-encoder (+0.23 on top).
See EXPERIMENT_LOG.md.
