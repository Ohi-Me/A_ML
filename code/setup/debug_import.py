import traceback, sys
for m in ["polars", "catboost"]:
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception:
        traceback.print_exc()
