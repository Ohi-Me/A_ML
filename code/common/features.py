"""Pair features for (S1, record) candidates. Country generic: no country one-hot, only comparisons.

No multiprocessing (named semaphores in /dev/shm disappear on the shared node): set features are polars list
expressions (multi-threaded Rust) and fuzzy scores use rapidfuzz cpdist with its own threads."""
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

WORKERS = -1


def _cp(x, y, scorer):
    return cpdist(x, y, scorer=scorer, workers=WORKERS, dtype=np.float32)


def _sets(x, y, pre):
    """x, y: list columns -> jaccard, containment of x in y (|x&y|/|x|), containment of y in x."""
    xu, yu = pl.col(x).list.unique(), pl.col(y).list.unique()
    inter = xu.list.set_intersection(yu).list.len().cast(pl.Float32)
    union = xu.list.set_union(yu).list.len().cast(pl.Float32)
    nx, ny = xu.list.len().cast(pl.Float32), yu.list.len().cast(pl.Float32)
    both = (nx > 0) & (ny > 0)
    return [
        pl.when(both).then(inter / union).otherwise(0.0).alias(f"{pre}_jac"),
        pl.when(both).then(inter / nx).otherwise(0.0).alias(f"{pre}_c12"),
        pl.when(both).then(inter / ny).otherwise(0.0).alias(f"{pre}_c21"),
    ]


LEGAL_FAMILY = {"inc": "corp", "corp": "corp", "co": "corp", "llc": "llc", "pllc": "prof", "pc": "prof", "pa": "prof",
                "ltd": "ltd", "pvt": "ltd", "plc": "ltd", "public": "ltd", "opc": "ltd", "llp": "part", "lp": "part",
                "sas": "fr_sas", "sasu": "fr_sas", "sarl": "fr_sarl", "eurl": "fr_sarl", "sa": "fr_sa", "sci": "fr_sci",
                "ei": "fr_ei", "snc": "fr_snc", "gmbh": "de", "ag": "de", "bv": "nl", "nv": "nl"}


def _code_nums(am):
    """digits of every address token that holds a digit: 'c-9-7' -> '97', '5-04' -> '504'."""
    return pl.col(am).str.split(" ").list.eval(
        pl.element().str.replace_all(r"[^0-9]", "").str.strip_chars_start("0").filter(pl.element() != ""))


def _token_align(c1, c2):
    """for each side: number of tokens with no fuzzy partner (ratio >= 80) on the other side, and the worst best-ratio."""
    n = len(c1)
    t1 = pl.DataFrame({"i": np.arange(n, dtype=np.int32), "t": c1}).with_columns(pl.col("t").str.split(" ")).explode("t")
    t2 = pl.DataFrame({"i": np.arange(n, dtype=np.int32), "u": c2}).with_columns(pl.col("u").str.split(" ")).explode("u")
    t1 = t1.filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
    t2 = t2.filter(pl.col("u").is_not_null() & (pl.col("u") != ""))
    j = t1.with_row_index("k1").join(t2.with_row_index("k2"), on="i")
    sc = cpdist(j["t"].to_list(), j["u"].to_list(), scorer=fuzz.ratio, workers=WORKERS, dtype=np.float32)
    j = j.with_columns(pl.Series("sc", sc))
    b1 = j.group_by(["i", "k1"]).agg(pl.col("sc").max()).group_by("i").agg(
        (pl.col("sc") < 80).sum().alias("tok_unm_12"), pl.col("sc").min().alias("tok_minbest_12"))
    b2 = j.group_by(["i", "k2"]).agg(pl.col("sc").max()).group_by("i").agg(
        (pl.col("sc") < 80).sum().alias("tok_unm_21"), pl.col("sc").min().alias("tok_minbest_21"))
    base = pl.DataFrame({"i": np.arange(n, dtype=np.int32)})
    out = base.join(b1, on="i", how="left").join(b2, on="i", how="left").sort("i")
    # a side with tokens but no pair at all (other side empty): every token is unmatched
    n1 = np.array([len([t for t in x.split(" ") if t]) for x in c1], np.float32)
    n2 = np.array([len([t for t in x.split(" ") if t]) for x in c2], np.float32)
    return pl.DataFrame({
        "tok_unm_12": out["tok_unm_12"].fill_null(0).to_numpy().astype(np.float32) + np.where(out["tok_minbest_12"].is_null().to_numpy(), n1, 0),
        "tok_unm_21": out["tok_unm_21"].fill_null(0).to_numpy().astype(np.float32) + np.where(out["tok_minbest_21"].is_null().to_numpy(), n2, 0),
        "tok_minbest_12": out["tok_minbest_12"].fill_null(0).to_numpy().astype(np.float32),
        "tok_minbest_21": out["tok_minbest_21"].fill_null(0).to_numpy().astype(np.float32),
    })


def compute_features_v2(s, r):
    """v1 features + number / legal-family / token-alignment features (F2)."""
    base = compute_features(s, r)
    f1, f2 = s["first_num"].to_list(), r["first_num"].to_list()
    lev = cpdist(f1, f2, scorer=distance.Levenshtein.distance, workers=WORKERS, dtype=np.float32)
    d = pl.DataFrame({"f1": s["first_num"], "f2": r["first_num"], "u1": s["nums"], "u2": r["nums"], "m1": s["am"],
                      "m2": r["am"], "l1": s["legal"], "l2": r["legal"]})
    fam = lambda c: pl.col(c).str.split(" ").list.eval(  # noqa: E731
        pl.element().replace_strict(LEGAL_FAMILY, default="").filter(pl.element() != "")).list.unique()
    d = d.with_columns(_code_nums("m1").alias("cn1"), _code_nums("m2").alias("cn2"), fam("l1").alias("fa1"),
                       fam("l2").alias("fa2"), pl.col("u1").str.split(" ").alias("uu1"), pl.col("u2").str.split(" ").alias("uu2"))
    both = (pl.col("f1") != "") & (pl.col("f2") != "")
    x1 = pl.col("f1").str.slice(0, 9).cast(pl.Float64, strict=False)
    x2 = pl.col("f2").str.slice(0, 9).cast(pl.Float64, strict=False)
    nfa1, nfa2 = pl.col("fa1").list.len(), pl.col("fa2").list.len()
    inter = pl.col("fa1").list.set_intersection(pl.col("fa2")).list.len()
    ex = d.select(
        pl.when(both).then(pl.Series(lev)).otherwise(-1.0).cast(pl.Float32).alias("fnum_lev"),
        pl.when(both).then((x1 - x2).abs() / pl.max_horizontal(x1, x2, pl.lit(1.0))).otherwise(-1.0).cast(pl.Float32)
        .alias("fnum_reldiff"),
        ((pl.col("f1") != "") & pl.col("uu2").list.contains(pl.col("f1"))).fill_null(False).cast(pl.Float32).alias("fnum1_in_r"),
        *_sets("cn1", "cn2", "cnum"),
        pl.col("cn2").list.unique().list.set_difference(pl.col("cn1")).list.len().cast(pl.Float32).alias("cnum_conflict"),
        pl.when((nfa1 > 0) & (nfa2 > 0) & (inter > 0)).then(1.0)
        .when((nfa1 > 0) & (nfa2 > 0)).then(-1.0).otherwise(0.0).cast(pl.Float32).alias("legal_fam_cmp"),
        ((nfa1 == 0) & (nfa2 > 0)).cast(pl.Float32).alias("legal_added"),
        ((nfa1 > 0) & (nfa2 == 0)).cast(pl.Float32).alias("legal_dropped"),
    )
    ta = _token_align(s["core"].to_list(), r["core"].to_list())
    return pl.concat([base, ex, ta], how="horizontal")


def compute_features(s, r):
    """s, r: normalized frames aligned row by row (s = S1 side, r = record side). Returns feature frame."""
    F = {}
    c1, c2 = s["core"].to_list(), r["core"].to_list()
    k1, k2 = s["key"].to_list(), r["key"].to_list()
    F["n_ratio"] = _cp(c1, c2, fuzz.ratio)
    F["n_tsort"] = _cp(c1, c2, fuzz.token_sort_ratio)
    F["n_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    F["n_partial"] = _cp(c1, c2, fuzz.partial_ratio)
    F["n_jw"] = _cp(c1, c2, distance.JaroWinkler.normalized_similarity)
    F["k_tsort"] = _cp(k1, k2, fuzz.token_sort_ratio)
    F["k_tset"] = _cp(k1, k2, fuzz.token_set_ratio)
    ns1 = [x.replace(" ", "") for x in c1]
    ns2 = [x.replace(" ", "") for x in c2]
    F["ns_ratio"] = _cp(ns1, ns2, fuzz.ratio)
    F["ns_partial"] = _cp(ns1, ns2, fuzz.partial_ratio)
    F["ns_lev"] = _cp(ns1, ns2, distance.Levenshtein.distance)
    del ns1, ns2
    am1, am2 = s["am"].to_list(), r["am"].to_list()
    F["a_tset"] = _cp(am1, am2, fuzz.token_set_ratio)
    F["a_tsort"] = _cp(am1, am2, fuzz.token_sort_ratio)
    F["a_partial"] = _cp(am1, am2, fuzz.partial_token_set_ratio)
    del c1, c2, k1, k2, am1, am2
    fz = pl.DataFrame(F)

    d = pl.DataFrame({
        "c1": s["core"], "c2": r["core"], "k1": s["key"], "k2": r["key"], "l1": s["legal"], "l2": r["legal"],
        "m1": s["am"], "m2": r["am"], "u1": s["nums"], "u2": r["nums"], "f1": s["first_num"], "f2": r["first_num"],
        "st1": s["state"], "st2": r["state"], "rid": r["id"], "dom": r["is_domain"], "nl": r["nonlatin"],
    })
    sp = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element() != ""))  # noqa: E731
    d = d.with_columns([sp(c).alias(f"t_{c}") for c in ("c1", "c2", "k1", "k2", "l1", "l2", "m1", "m2", "u1", "u2")])
    out = d.select(
        *_sets("t_c1", "t_c2", "n"),
        *_sets("t_k1", "t_k2", "k"),
        (pl.col("k1") == pl.col("k2")).cast(pl.Float32).alias("k_eq"),
        (pl.col("t_k1").list.sort().list.join(" ") == pl.col("t_k2").list.sort().list.join(" ")).cast(pl.Float32)
        .alias("k_sorted_eq"),
        ((pl.col("t_c1").list.first() == pl.col("t_c2").list.first()).fill_null(False)).cast(pl.Float32).alias("first_tok_eq"),
        pl.col("c1").str.len_chars().cast(pl.Float32).alias("n_len1"),
        pl.col("c2").str.len_chars().cast(pl.Float32).alias("n_len2"),
        pl.col("t_c1").list.len().cast(pl.Float32).alias("n_ntok1"),
        pl.col("t_c2").list.len().cast(pl.Float32).alias("n_ntok2"),
        (pl.col("l1") != "").cast(pl.Float32).alias("legal_1"),
        (pl.col("l2") != "").cast(pl.Float32).alias("legal_2"),
        _sets("t_l1", "t_l2", "legal")[0],
        pl.col("dom").cast(pl.Float32).alias("is_domain"),
        pl.col("nl").cast(pl.Float32).alias("nonlatin"),
        pl.col("c2").str.contains(r"[^\x00-\x7f]").cast(pl.Float32).alias("nonascii_left"),
        pl.col("rid").str.starts_with("S3").cast(pl.Float32).alias("src3"),
        (pl.col("m2") == "").cast(pl.Float32).alias("a_empty2"),
        *_sets("t_m1", "t_m2", "a"),
        pl.col("t_m2").list.len().cast(pl.Float32).alias("a_ntok2"),
        *_sets("t_u1", "t_u2", "num"),
        pl.col("t_u1").list.len().cast(pl.Float32).alias("num_n1"),
        pl.col("t_u2").list.len().cast(pl.Float32).alias("num_n2"),
        pl.col("t_u2").list.unique().list.set_difference(pl.col("t_u1")).list.len().cast(pl.Float32).alias("num_conflict"),
        ((pl.col("f1") != "") & (pl.col("f1") == pl.col("f2"))).cast(pl.Float32).alias("fnum_eq"),
        ((pl.col("f2") != "") & pl.col("t_u1").list.contains(pl.col("f2"))).fill_null(False).cast(pl.Float32).alias("fnum_in"),
        pl.when((pl.col("st1") == "") | (pl.col("st2") == "")).then(0.0)
        .when(pl.col("st1") == pl.col("st2")).then(1.0).otherwise(-1.0).cast(pl.Float32).alias("state_cmp"),
    )
    return pl.concat([fz, out], how="horizontal")
