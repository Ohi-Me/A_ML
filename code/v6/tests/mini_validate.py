"""Smoke-test stand-in for the official validator (which needs the challenge TSVs): checks the two submission files
of a folder - header, one row per test S1 in file order, known S2/S3 ids only, no duplicates, matched within candidates.
  python code/v6/tests/mini_validate.py results/final/<name>/output"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common.decide import ids  # noqa: E402

out = sys.argv[1]
s1, rr = ids("test")
known = set(rr["id"].to_list())
lists = {}
for fn, col in (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids")):
    d = pl.read_csv(os.path.join(out, fn), separator="\t", quote_char=None, infer_schema=False,
                    missing_utf8_is_empty_string=True)
    assert d.columns == ["source1_entity_id", col], d.columns
    assert d["source1_entity_id"].to_list() == s1["id"].to_list(), "one row per test S1, file order"
    rows = [[x for x in v.split(",") if x] for v in d[col].fill_null("").to_list()]
    for xs in rows:
        assert len(xs) == len(set(xs)), "duplicate id in a list"
        assert all(x in known for x in xs), "unknown id"
    lists[col] = rows
assert all(set(m) <= set(c) for m, c in zip(lists["matched_entity_ids"], lists["candidate_entity_ids"]))
print("mini validator PASS", out, "matches", sum(len(m) for m in lists["matched_entity_ids"]))
