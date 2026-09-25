"""Label-free structural diff of the submitted matching files (no data or GPU needed; runs on a laptop).

Which pairs does V3 accept that V2 does not, and where do they land (new entities vs extra records on entities
V2 already matched)? Output: results/audit/submission_diff.json

  python code/audit/submission_diff.py
"""
import json
import os
import sys

import polars as pl


def load(n):
    d = pl.read_csv(f"submissions/{n}/matching_results.tsv", separator="\t", quote_char=None, infer_schema=False) \
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "m"}).with_columns(pl.col("m").fill_null(""))
    per = d.with_columns(pl.when(pl.col("m")=="").then(0).otherwise(pl.col("m").str.count_matches(",")+1).alias("k"))
    pairs = d.filter(pl.col("m")!="").with_columns(pl.col("m").str.split(",")).explode("m").rename({"m":"r"})
    return per.select(["s1","k"]), pairs


out = {}
S = {n: load(n) for n in ["final_c1","final_c2v1","final_v2","final_v3"]}
for n,(per,pairs) in S.items():
    kc = per["k"].value_counts().sort("k")
    out[n] = {"pairs": pairs.height, "empty_s1": int((per["k"]==0).sum()), "empty_rate": float((per["k"]==0).mean()),
              "mean_k": float(per["k"].mean()), "k_hist": {int(a):int(b) for a,b in kc.iter_rows()},
              "records_S2": int(pairs["r"].str.starts_with("S2").sum()), "records_S3": int(pairs["r"].str.starts_with("S3").sum())}
def diff(x, y):
    (px, Px), (py, Py) = S[x], S[y]
    only_y = Py.join(Px, on=["s1","r"], how="anti"); only_x = Px.join(Py, on=["s1","r"], how="anti")
    common = Py.join(Px, on=["s1","r"], how="semi").height
    # records present in both but assigned to a different S1
    ry = only_y.join(Px.rename({"s1":"s1x"}), on="r", how="left")
    moved = int(ry["s1x"].is_not_null().sum())
    kk = px.rename({"k":"kx"}).join(py.rename({"k":"ky"}), on="s1")
    trans = kk.filter(pl.col("kx")!=pl.col("ky")).with_columns(
        pl.when(pl.col("kx")==0).then(pl.lit("empty->nonempty")).when(pl.col("ky")==0).then(pl.lit("nonempty->empty"))
        .when(pl.col("ky")>pl.col("kx")).then(pl.lit("grow")).otherwise(pl.lit("shrink")).alias("t"))
    tc = {t:int(c) for t,c in trans["t"].value_counts().iter_rows()}
    # where do y's extra pairs land: S1s y made non-empty vs S1s already matched in x
    oy = only_y.join(kk, on="s1")
    return {"common": common, f"only_{y}": only_y.height, f"only_{x}": only_x.height, "net": only_y.height-only_x.height,
            f"only_{y}_record_was_assigned_elsewhere_in_{x}": moved,
            f"only_{y}_record_new (unassigned in {x})": only_y.height-moved,
            f"only_{y}_on_S1_empty_in_{x}": int((oy["kx"]==0).sum()),
            f"only_{y}_by_kx": {int(a):int(b) for a,b in oy["kx"].value_counts().sort("kx").iter_rows()},
            "s1_k_transitions": tc, "s1_changed": trans.height,
            "jaccard_pairs": common/(common+only_y.height+only_x.height)}
out["diff_v2_to_v3"] = diff("final_v2","final_v3")
out["diff_c2v1_to_v2"] = diff("final_c2v1","final_v2")
out["diff_c1_to_v2"] = diff("final_c1","final_v2")
print(json.dumps(out, indent=1))
os.makedirs("results/audit", exist_ok=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "results/audit/submission_diff.json", "w"), indent=1)
