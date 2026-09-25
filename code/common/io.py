"""Reading the challenge files. TSV has no quoting (the validator just splits on tab), so we read the same way.
Raw files are cached once as parquet under data/cache/ (a copy; the original TSVs are never changed)."""
import os

import polars as pl

ROOT = os.environ.get("ER_ROOT", os.getcwd())
DATA = os.path.join(ROOT, "data", "dataset")
RAW_CACHE = os.path.join(ROOT, "data", "cache")          # raw parquet copies + ground truth (shared)
NORM = os.environ.get("ER_NORM", "v1")                    # normalization version
CACHE = os.environ.get("ER_CACHE", RAW_CACHE)             # derived artifacts of this normalization version
RESULTS = os.path.join(ROOT, "results")


def _read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, has_header=True,
                       missing_utf8_is_empty_string=True, truncate_ragged_lines=False)


def load_source(split, k):
    """split in {'train','test'}, k in {1,2,3}. Returns polars DataFrame with string columns."""
    os.makedirs(RAW_CACHE, exist_ok=True)
    pq = os.path.join(RAW_CACHE, f"{split}_source{k}.parquet")
    if os.path.exists(pq):
        return pl.read_parquet(pq)
    df = _read_tsv(os.path.join(DATA, split, f"{split}_source{k}.tsv"))
    df = df.with_columns([pl.col(c).fill_null("") for c in df.columns])
    df.write_parquet(pq)
    return df


def load_gt():
    """Ground truth as (long pairs, per-S1 table). Long pairs: s1, rid. Per S1: s1, matches(list)."""
    os.makedirs(RAW_CACHE, exist_ok=True)
    pq = os.path.join(RAW_CACHE, "train_gt_pairs.parquet")
    pq2 = os.path.join(RAW_CACHE, "train_gt.parquet")
    if os.path.exists(pq) and os.path.exists(pq2):
        return pl.read_parquet(pq), pl.read_parquet(pq2)
    g = _read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv")).with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    g = g.with_columns(pl.when(pl.col("matched_entity_ids") == "").then(pl.lit([], dtype=pl.List(pl.Utf8)))
                       .otherwise(pl.col("matched_entity_ids").str.split(",")).alias("matches"))
    per = g.select(pl.col("source1_entity_id").alias("s1"), "matches")
    pairs = per.explode("matches").rename({"matches": "rid"}).filter(pl.col("rid").is_not_null() & (pl.col("rid") != ""))
    pairs.write_parquet(pq)
    per.write_parquet(pq2)
    return pairs, per


def write_id_lists(path, s1_ids, lists, col):
    """Write matching_results.tsv / candidate_pairs.tsv. lists: dict s1 -> list of ids (deduped, order kept)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in s1_ids:
            ids = list(dict.fromkeys(lists.get(s, [])))
            fh.write(f"{s}\t{','.join(ids)}\n")
