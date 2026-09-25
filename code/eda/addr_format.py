"""Address / name format statistics per split, source and country: is test formatted like train?"""
import json
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.io import RESULTS, load_source  # noqa: E402

gpu_init()
out = {}
for sp in ("train", "test"):
    for k in (1, 2, 3):
        d = load_source(sp, k).sample(300_000, seed=0)
        a = pl.col("business_address")
        s = d.group_by("country").agg(
            pl.len().alias("n"),
            a.str.contains(r"^\s*\d").mean().alias("starts_digit"),
            a.str.count_matches(",").mean().alias("commas"),
            a.str.contains(r"(?i)\bunit\b").mean().alias("has_unit"),
            a.str.contains(r"(?i)\b(pmb|po box)\b").mean().alias("has_pmb_po"),
            a.str.contains(r"(?i)\bfl\b|floor").mean().alias("has_floor"),
            a.str.contains(r"\d+\s*-\s*\d+").mean().alias("num_range"),
            a.str.len_chars().mean().alias("addr_len"),
            (a.str.strip_chars() == "").mean().alias("empty"),
            a.str.contains(r"^[A-Z]{2},").mean().alias("starts_state_code"),
            a.str.contains(r",\s*[A-Z]{2}\s*$").mean().alias("ends_state_code"),
            pl.col("business_name").str.len_chars().mean().alias("name_len"),
        ).sort("country")
        for row in s.iter_rows(named=True):
            key = f"{sp}_s{k}_{row['country']}"
            out[key] = {c: (round(v, 4) if isinstance(v, float) else v) for c, v in row.items() if c != "country"}
            print(key, out[key], flush=True)
json.dump(out, open(os.path.join(RESULTS, "eda", "addr_format.json"), "w"), indent=1)
