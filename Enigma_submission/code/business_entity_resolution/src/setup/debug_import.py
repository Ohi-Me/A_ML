# Team Enigma (Rohit Kumar, Vishesh Shekhawat, Sriyansh, Veeky Kumar) - ML Challenge 2026, Business Entity Resolution
import traceback, sys
for m in ["polars", "catboost"]:
    try:
        mod = __import__(m); print(m, getattr(mod, "__version__", "?"))
    except Exception:
        traceback.print_exc()
