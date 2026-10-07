# Team Enigma (Rohit Kumar, Vishesh Shekhawat, Sriyansh, Veeky Kumar) - ML Challenge 2026, Business Entity Resolution
import ssl
import urllib.request
from huggingface_hub import hf_hub_download

ssl._create_default_https_context = ssl._create_unverified_context

try:
    code = urllib.request.urlopen("https://huggingface.co", timeout=5).getcode()
    print(f"HF reachability with unverified SSL: OK (status {code})")
    
    # Try downloading tokenizer config of bge-reranker-v2-m3
    path = hf_hub_download(repo_id="BAAI/bge-reranker-v2-m3", filename="tokenizer_config.json")
    print(f"HF Download OK: {path}")
except Exception as e:
    print(f"HF Download FAIL: {e}")
