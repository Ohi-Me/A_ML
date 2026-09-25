"""Check normalization v1 on real true pairs and measure speed."""
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import gpu_init  # noqa: E402
from common.normalize import addr_tokens_for_match, norm_addr, norm_name, to_ascii  # noqa: E402

gpu_init()
OUT = os.path.join("results", "eda")
tp = pl.read_csv(os.path.join(OUT, "true_pairs_sample.tsv"), separator="\t", quote_char=None, infer_schema=False)
lines = []
for s1, rid, c1, c2, n1, n2, a1, a2 in tp.head(150).iter_rows():
    x1, x2 = norm_name(n1), norm_name(n2)
    y1, y2 = norm_addr(a1), norm_addr(a2)
    lines.append(f"{rid} [{c1}]\n  N1 {n1!r} -> core={x1['core']!r} key={x1['key']!r} legal={x1['legal']!r}\n"
                 f"  N2 {n2!r} -> core={x2['core']!r} key={x2['key']!r} legal={x2['legal']!r} dom={x2['is_domain']}\n"
                 f"  A1 {a1!r} -> {y1['a']!r} nums={y1['nums']!r} st={y1['state']!r} m={addr_tokens_for_match(y1['a'])}\n"
                 f"  A2 {a2!r} -> {y2['a']!r} nums={y2['nums']!r} st={y2['state']!r} m={addr_tokens_for_match(y2['a'])}\n")
indic = tp.filter(pl.col("n2").str.contains(r"[ऀ-෿]")).head(40)
for n1, n2, a2 in indic.select(["n1", "n2", "a2"]).iter_rows():
    lines.append(f"INDIC {n1!r} <- {n2!r} => {to_ascii(n2)!r} | core {norm_name(n2)['core']!r} | addr {to_ascii(a2)!r}")
names = tp["n2"].to_list() * 30
addrs = tp["a2"].to_list() * 30
t = time.time()
for n in names:
    norm_name(n)
tn = (time.time() - t) / len(names) * 1e6
t = time.time()
for a in addrs:
    norm_addr(a)
ta = (time.time() - t) / len(addrs) * 1e6
lines.append(f"speed: name {tn:.1f} us/rec, addr {ta:.1f} us/rec")
open(os.path.join(OUT, "norm_check_v1.txt"), "w", encoding="utf-8").write("\n".join(lines))
print(lines[-1])
