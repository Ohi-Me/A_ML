"""Blocking keys shared by the V7 probe and candidate builder (so both measure and build the same thing)."""
import polars as pl


# number markers and suffixes that survive normalisation ("N°" -> "ndeg", "No.", "bis", "ter") and would make the same
# street look different ("ndeg 12 rue x" vs "12 rue x")
NUM_WORDS = ["ndeg", "deg", "no", "nr", "num", "n", "bis", "ter", "quater", "b", "a", "c"]


def street_core(am):
    """normalised address without any token that contains a digit or is a number marker, tokens sorted ('' if empty)."""
    return pl.Series(am).fill_null("").str.replace_all(r"\S*\d\S*", "").str.split(" ").list.eval(
        pl.element().filter((pl.element() != "") & ~pl.element().is_in(NUM_WORDS))).list.sort().list.join(" ")


def numstreet_key(am, first_num):
    """first house number + street core; '' when either is missing."""
    s = street_core(am)
    n = pl.Series(first_num).fill_null("")
    return pl.select(pl.when((n != "") & (s != "")).then(pl.concat_str([n, s], separator="|")).otherwise(pl.lit("")))\
        .to_series().to_numpy()
