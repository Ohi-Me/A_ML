"""Stage prep: learn the transliteration maps (training folds only) and write normalized parquet for one split.

  python code/common/prep.py --split train --dict folds   # for validation work: maps learned without fold 0
  python code/common/prep.py --split test  --dict all     # for the final run: maps learned on all training pairs
"""
import argparse
import json
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.join(os.getcwd(), "code"))
from common.gpu import Timer, gpu_init  # noqa: E402
from common.io import CACHE, NORM, RESULTS, load_gt, load_source  # noqa: E402
from common.split import VAL_FOLD, fold_of  # noqa: E402
from common import normalize as N  # noqa: E402
from common import translit  # noqa: E402

VERSION = NORM
N.ADMIN_V2 = NORM != "v1"     # v2: French regions / departments as state, whole-part state names
COLS = ["n", "core", "key", "legal", "is_domain", "nonlatin", "a", "am", "nums", "state", "first_num", "comps"]
SCHEMA = {c: (pl.Boolean if c in ("is_domain", "nonlatin") else (pl.Int16 if c == "comps" else pl.Utf8)) for c in COLS}


def _init(words, states):
    N.set_translit(words, states)
    N.ADMIN_V2 = NORM != "v1"


def _work(chunk):
    names, addrs = chunk
    cols = {c: [] for c in COLS}
    for nm, ad in zip(names, addrs):
        x = N.norm_name(nm)
        y = N.norm_addr(ad)
        for c, v in (("n", x["n"]), ("core", x["core"]), ("key", x["key"]), ("legal", x["legal"]),
                     ("is_domain", x["is_domain"]), ("nonlatin", x["nonlatin"]), ("a", y["a"]),
                     ("am", " ".join(N.addr_tokens_for_match(y["a"]))), ("nums", y["nums"]), ("state", y["state"]),
                     ("first_num", y["first_num"]), ("comps", y["comps"])):
            cols[c].append(v)
    return pl.DataFrame(cols, schema=SCHEMA)   # pickled as arrow: compact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--dict", default="folds", choices=["folds", "all"])
    args = ap.parse_args()
    gpu_init()
    out_dir = os.path.join(RESULTS, "eda")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    dict_path = os.path.join(CACHE, f"translit_{args.dict}.json")

    if not os.path.exists(dict_path):
        with Timer(f"learn translit maps ({args.dict})"):
            pairs, _ = load_gt()
            s1 = load_source("train", 1).rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1"})
            rr = pl.concat([load_source("train", 2), load_source("train", 3)]).rename(
                {"entity_id": "rid", "business_name": "n2", "business_address": "a2"})
            rr = rr.filter(pl.col("n2").str.contains(r"[^\x00-ɏ\s]") | pl.col("a2").str.contains(r"[^\x00-ɏ\s]"))
            p = pairs.join(rr.select(["rid", "n2", "a2"]), on="rid").join(s1.select(["s1", "n1", "a1"]), on="s1")
            if args.dict == "folds":
                p = p.filter(pl.col("s1").map_elements(fold_of, return_dtype=pl.Int64) != VAL_FOLD)
            words, states = translit.learn(p.select(["n1", "a1", "n2", "a2"]).iter_rows())
            translit.save(dict_path, words, states)
            print(f"pairs used {p.height}, word map {len(words)}, state map {len(states)}", flush=True)
            del p, rr, s1, pairs
    words, states = translit.load(dict_path)
    N.set_translit(words, states)

    stats = {}
    # one process on purpose: named semaphores in /dev/shm vanish on this node and hang worker pools
    os.makedirs(CACHE, exist_ok=True)
    for k in (1, 2, 3):
        path = os.path.join(CACHE, f"norm_{VERSION}_{args.split}_s{k}.parquet")
        df = load_source(args.split, k)
        with Timer(f"normalize {args.split} s{k} ({df.height} rows)"):
            step = 200000
            nd = pl.concat([_work((df["business_name"][i:i + step].to_list(), df["business_address"][i:i + step].to_list()))
                            for i in range(0, df.height, step)])
            nd = pl.concat([df.select(pl.col("entity_id").alias("id"), "country"), nd], how="horizontal")
            nd.write_parquet(path)
        nl = nd.filter(pl.col("nonlatin"))
        stats[f"s{k}"] = {"rows": nd.height, "nonlatin_names": nl.height, "empty_addr": int((nd["a"] == "").sum()),
                          "no_state": int((nd["state"] == "").sum()), "domain": int(nd["is_domain"].sum()),
                          "empty_core": int((nd["core"] == "").sum()),
                          "state_counts_top": dict(nd["state"].value_counts().sort("count", descending=True).head(30).iter_rows())}
        print({k2: v for k2, v in stats[f"s{k}"].items() if k2 != "state_counts_top"}, flush=True)
        del df, nd
    stats["translit_words"] = len(words)
    stats["translit_states"] = len(states)
    json.dump(stats, open(os.path.join(out_dir, f"prep_{VERSION}_{args.split}_{args.dict}.json"), "w"), indent=1)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"total {time.time() - t:.0f}s")
