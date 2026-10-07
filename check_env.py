import torch
import xgboost
import polars
import sklearn
import scipy
import rapidfuzz
import anyascii
import transformers

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1024**3)
print("XGBoost version:", xgboost.__version__)
print("Polars version:", polars.__version__)
