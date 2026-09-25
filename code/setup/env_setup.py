"""Install the few missing packages into RDIR/pkgs (the shared conda env is not touched) and check the GPU stack.
Runs inside a PBS GPU job."""
import json
import os
import subprocess
import sys
import time

ROOT = os.getcwd()
PKGS = os.path.join(ROOT, "pkgs")
OUT = os.path.join(ROOT, "results", "setup")
os.makedirs(PKGS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# no-deps on purpose: torch / numpy / transformers stay the env versions
NODEPS = ["rapidfuzz", "polars", "polars-runtime-32", "anyascii", "jellyfish", "catboost",
          "sentence-transformers", "graphviz", "narwhals"]
if "--skip-install" not in sys.argv:
    for p in NODEPS:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "--upgrade",
                            "--target", PKGS, p], capture_output=True, text=True)
        print(p, "rc", r.returncode, (r.stderr or "")[-400:] if r.returncode else "", flush=True)

sys.path.insert(0, PKGS)
report = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version}
import torch  # noqa: E402

report["torch"] = torch.__version__
report["torch_cuda"] = torch.version.cuda
report["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    report["gpu"] = p.name
    report["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    report["fp16_tflops"] = round(20 * 2 * 8192**3 / (time.time() - t) / 1e12, 1)

for mod in ["numpy", "pandas", "pyarrow", "polars", "rapidfuzz", "anyascii", "jellyfish", "sklearn", "scipy",
            "lightgbm", "xgboost", "catboost", "transformers", "sentence_transformers", "optuna"]:
    try:
        m = __import__(mod)
        report[mod] = getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        report[mod] = f"FAIL {type(e).__name__}: {e}"

import numpy as np  # noqa: E402

X = np.random.rand(20000, 20).astype("float32")
y = (X[:, 0] + X[:, 1] > 1).astype(int)
try:
    import xgboost as xgb
    b = xgb.train({"tree_method": "hist", "device": "cuda", "objective": "binary:logistic"}, xgb.DMatrix(X, y), 20)
    report["xgboost_cuda"] = "ok"
except Exception as e:  # noqa: BLE001
    report["xgboost_cuda"] = f"FAIL {e}"
try:
    from catboost import CatBoostClassifier
    CatBoostClassifier(iterations=20, task_type="GPU", verbose=0).fit(X, y)
    report["catboost_gpu"] = "ok"
except Exception as e:  # noqa: BLE001
    report["catboost_gpu"] = f"FAIL {e}"
try:
    import lightgbm as lgb
    lgb.train({"objective": "binary", "device_type": "cuda", "verbose": -1}, lgb.Dataset(X, y), 5)
    report["lightgbm_cuda"] = "ok"
except Exception as e:  # noqa: BLE001
    report["lightgbm_cuda"] = f"FAIL {str(e)[:200]}"
try:
    import lightgbm as lgb
    lgb.train({"objective": "binary", "device_type": "gpu", "verbose": -1}, lgb.Dataset(X, y), 5)
    report["lightgbm_opencl"] = "ok"
except Exception as e:  # noqa: BLE001
    report["lightgbm_opencl"] = f"FAIL {str(e)[:200]}"

report["ncpus_job"] = os.environ.get("NCPUS")
report["cpu_count"] = os.cpu_count()
print(json.dumps(report, indent=1))
json.dump(report, open(os.path.join(OUT, "env_report.json"), "w"), indent=1)
