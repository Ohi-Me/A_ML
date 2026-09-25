"""Stage 1: look at the real data before building anything.
Writes results/eda/eda_report.json, eda_report.md and sample files of true pairs for reading."""
import json
import os
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import RESULTS, load_gt, load_source  # noqa: E402

OUT = os.path.join(RESULTS, "eda")
os.makedirs(OUT, exist_ok=True)
gpu_init()
R = {}
md = ["# EDA report", ""]


def script_of(s):
    for ch in s:
        if ord(ch) > 127 and ch.isalpha():
            try:
                return unicodedata.name(ch).split()[0]
            except ValueError:
                return "OTHER"
    return "LATIN_ASCII"


with Timer("load"):
    src = {(sp, k): load_source(sp, k) for sp in ("train", "test") for k in (1, 2, 3)}
    pairs, per = load_gt()

# ---------------------------------------------------------------- basic shape
for (sp, k), df in src.items():
    key = f"{sp}_s{k}"
    d = {"rows": df.height, "cols": df.columns,
         "dup_ids": int(df.height - df["entity_id"].n_unique()),
         "dup_name_addr_country": int(df.height - df.select(["business_name", "business_address", "country"]).n_unique()),
         "bad_prefix": int(df.filter(~pl.col("entity_id").str.starts_with(f"S{k}-")).height)}
    for c in ("business_name", "business_address", "country"):
        s = df[c]
        d[f"empty_{c}"] = int((s.str.strip_chars() == "").sum())
        d[f"null_literal_{c}"] = int(s.str.to_lowercase().str.contains(r"\bnull\b|\bnan\b|\bnone\b").sum())
    d["country_counts"] = dict(df["country"].value_counts().sort("count", descending=True).iter_rows())
    d["name_len_mean"] = float(df["business_name"].str.len_chars().mean())
    d["addr_len_mean"] = float(df["business_address"].str.len_chars().mean())
    samp = df.sample(min(200000, df.height), seed=0)
    d["name_script"] = dict(Counter(script_of(x) for x in samp["business_name"]).most_common(12))
    d["addr_script"] = dict(Counter(script_of(x) for x in samp["business_address"]).most_common(12))
    R[key] = d
    print(key, json.dumps({k2: v for k2, v in d.items() if k2 not in ("cols",)}, ensure_ascii=False)[:1500], flush=True)

# ---------------------------------------------------------------- ground truth structure
g = {}
s1_train = set(src[("train", 1)]["entity_id"].to_list())
gt_s1 = set(per["s1"].to_list())
g["gt_rows"] = per.height
g["gt_s1_unique"] = len(gt_s1)
g["s1_in_file_not_in_gt"] = len(s1_train - gt_s1)
g["gt_s1_not_in_file"] = len(gt_s1 - s1_train)
nm = per.with_columns(pl.col("matches").list.len().alias("n"))
g["singletons"] = int((nm["n"] == 0).sum())
g["singleton_rate"] = g["singletons"] / per.height
g["matches_per_s1_hist"] = {int(a): int(b) for a, b in nm["n"].value_counts().sort("n").iter_rows()}
pairs = pairs.with_columns(pl.col("rid").str.slice(0, 2).alias("src"))
g["pairs_total"] = pairs.height
g["pairs_by_src"] = dict(pairs["src"].value_counts().iter_rows())
per_src = pairs.group_by(["s1", "src"]).len().rename({"len": "k"})
g["per_s1_per_src_hist"] = {f"{a}:{b}": int(c) for a, b, c in
                            per_src.group_by(["src", "k"]).len().sort(["src", "k"]).iter_rows()}
rid_multi = pairs.group_by("rid").len().filter(pl.col("len") > 1)
g["rids_matched_to_multiple_s1"] = rid_multi.height
all_rids = set(src[("train", 2)]["entity_id"].to_list()) | set(src[("train", 3)]["entity_id"].to_list())
matched_rids = set(pairs["rid"].to_list())
g["rids_not_in_files"] = len(matched_rids - all_rids)
g["s2_matched_frac"] = len(matched_rids & set(src[("train", 2)]["entity_id"].to_list())) / src[("train", 2)].height
g["s3_matched_frac"] = len(matched_rids & set(src[("train", 3)]["entity_id"].to_list())) / src[("train", 3)].height
R["gt"] = g
print("GT", json.dumps(g)[:3000], flush=True)

# ---------------------------------------------------------------- joined true pairs
s1 = src[("train", 1)].rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1", "country": "c1"})
rr = pl.concat([src[("train", 2)], src[("train", 3)]]).rename(
    {"entity_id": "rid", "business_name": "n2", "business_address": "a2", "country": "c2"})
tp = pairs.join(s1, on="s1", how="left").join(rr, on="rid", how="left")
tp = tp.with_columns([
    (pl.col("c1") == pl.col("c2")).alias("same_country"),
    (pl.col("n1").str.to_lowercase() == pl.col("n2").str.to_lowercase()).alias("same_name_lc"),
    (pl.col("a1").str.to_lowercase() == pl.col("a2").str.to_lowercase()).alias("same_addr_lc"),
])
cc = {}
for src_ in ("S2", "S3"):
    t = tp.filter(pl.col("src") == src_)
    cc[src_] = {"same_country": float(t["same_country"].mean()), "same_name_lc": float(t["same_name_lc"].mean()),
                "same_addr_lc": float(t["same_addr_lc"].mean()),
                "country_pairs": {f"{a}->{b}": int(c) for a, b, c in t.group_by(["c1", "c2"]).len().sort("len", descending=True).head(10).iter_rows()}}
R["true_pairs"] = cc
print("TRUE PAIRS", json.dumps(cc, ensure_ascii=False), flush=True)

# digit agreement in true pairs (house numbers, PIN / ZIP)
def digs(s):
    return set(re.findall(r"\d+", s or ""))


samp = tp.sample(min(300000, tp.height), seed=1)
agree = Counter()
for a1, a2, src_, c1 in samp.select(["a1", "a2", "src", "c1"]).iter_rows():
    d1, d2 = digs(a1), digs(a2)
    k = f"{src_}_{c1}"
    agree[k + "_n"] += 1
    agree[k + "_both_have_digits"] += bool(d1 and d2)
    agree[k + "_share_digit"] += bool(d1 & d2)
    agree[k + "_a2_digits_subset"] += bool(d2 and d2 <= d1)
    agree[k + "_a2_empty"] += not (a2 or "").strip()
R["digit_agreement_true_pairs"] = dict(agree)

with open(os.path.join(OUT, "true_pairs_sample.tsv"), "w", encoding="utf-8") as fh:
    fh.write("s1\trid\tc1\tc2\tn1\tn2\ta1\ta2\n")
    for row in tp.sample(3000, seed=2).select(["s1", "rid", "c1", "c2", "n1", "n2", "a1", "a2"]).iter_rows():
        fh.write("\t".join(str(x).replace("\t", " ") for x in row) + "\n")

# a few S1 entities with all of their matches, to see the multi-match structure
big = nm.filter(pl.col("n") >= 4).sample(40, seed=3)["s1"].to_list()
with open(os.path.join(OUT, "multi_match_examples.txt"), "w", encoding="utf-8") as fh:
    for s in big:
        r1 = s1.filter(pl.col("s1") == s).row(0)
        fh.write(f"\n### {r1[0]} | {r1[1]} | {r1[2]} | {r1[3]}\n")
        for row in tp.filter(pl.col("s1") == s).select(["rid", "n2", "a2", "c2"]).iter_rows():
            fh.write(f"   {row[0]} | {row[1]} | {row[2]} | {row[3]}\n")

# singletons: what do they look like
sing = nm.filter(pl.col("n") == 0).join(s1, on="s1").sample(300, seed=4)
sing.select(["s1", "n1", "a1", "c1"]).write_csv(os.path.join(OUT, "singletons_sample.tsv"), separator="\t")

# unmatched S2/S3 records sample
un = rr.filter(~pl.col("rid").is_in(list(matched_rids))).sample(300, seed=5)
un.write_csv(os.path.join(OUT, "unmatched_rids_sample.tsv"), separator="\t")
R["unmatched_rids"] = int(rr.height - len(matched_rids & all_rids))

# ---------------------------------------------------------------- token statistics (legal suffixes, noise)
tok = Counter()
first_tok = Counter()
last_tok = Counter()
for (sp, k) in [("train", 1), ("train", 2), ("train", 3), ("test", 1)]:
    ss = src[(sp, k)].sample(100000, seed=6)["business_name"].to_list()
    for s in ss:
        t = s.split()
        if t:
            first_tok[(sp, k, t[0])] += 1
            last_tok[(sp, k, t[-1])] += 1
        for w in t:
            tok[(sp, k, w)] += 1
tops = {}
for (sp, k) in [("train", 1), ("train", 2), ("train", 3), ("test", 1)]:
    tops[f"{sp}_s{k}_top_tokens"] = [(w, c) for (a, b, w), c in tok.most_common(4000) if (a, b) == (sp, k)][:60]
    tops[f"{sp}_s{k}_first_tokens"] = [(w, c) for (a, b, w), c in first_tok.most_common(4000) if (a, b) == (sp, k)][:40]
    tops[f"{sp}_s{k}_last_tokens"] = [(w, c) for (a, b, w), c in last_tok.most_common(4000) if (a, b) == (sp, k)][:40]
R["tokens"] = tops

# non-alnum characters used in names per source
chars = Counter()
for (sp, k) in [("train", 1), ("train", 2), ("train", 3)]:
    for s in src[(sp, k)].sample(100000, seed=7)["business_name"].to_list():
        for ch in s:
            if not ch.isalnum() and ch != " ":
                chars[(k, ch)] += 1
R["name_punct"] = {f"s{k}": [(ch, c) for (kk, ch), c in chars.most_common(200) if kk == k][:30] for k in (1, 2, 3)}

# test-only: France examples
fr = {}
for k in (1, 2, 3):
    df = src[("test", k)]
    f = df.filter(pl.col("country") == "France")
    fr[f"s{k}"] = f.height
    f.sample(min(50, f.height), seed=8).write_csv(os.path.join(OUT, f"test_france_s{k}_sample.tsv"), separator="\t")
R["test_france_rows"] = fr

json.dump(R, open(os.path.join(OUT, "eda_report.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("DONE", flush=True)
