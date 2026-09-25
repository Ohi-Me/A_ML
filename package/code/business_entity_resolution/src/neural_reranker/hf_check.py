"""Can the job fetch and run a pretrained multilingual encoder (MIT licence) on the GPU?"""
import time

import torch
from transformers import AutoModel, AutoTokenizer

from common.gpu import gpu_init

dev = gpu_init()
name = "FacebookAI/xlm-roberta-base"
t = time.time()
tok = AutoTokenizer.from_pretrained(name)
model = AutoModel.from_pretrained(name).to(dev)
print("loaded", name, sum(p.numel() for p in model.parameters()) / 1e6, "M params in", round(time.time() - t), "s")
x = tok(["Garoa | 313 Berry Avenue, Ashland, KY"] * 256, ["GAROA CORP | 313 BERRY AVE, ASHLAND, KY"] * 256, padding=True,
        truncation=True, max_length=128, return_tensors="pt").to(dev)
print(tok.convert_ids_to_tokens(x["input_ids"][0][:40]))
with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
    torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        model(**x)
    torch.cuda.synchronize()
print("inference pairs/s", round(2560 / (time.time() - t)))
print(tok.tokenize("ऑल सिस्टम्स प्राइवेट लिमिटेड | Société Ets Raid SAS, 287 Avenue Linné, Roubaix"))
