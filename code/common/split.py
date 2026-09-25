"""Entity level folds. A Source-1 entity and all of its true matches always sit in the same fold.
Fold 0 is the validation fold; folds 1-4 are for fitting. Unmatched S2/S3 records get fold -1."""
import zlib

import polars as pl

N_FOLDS = 5
VAL_FOLD = 0


def fold_of(s1_id):
    return zlib.crc32(s1_id.encode()) % N_FOLDS


def fold_expr(col="s1"):
    return pl.col(col).map_elements(fold_of, return_dtype=pl.Int8)
