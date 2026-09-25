#!/bin/bash
# Build the final submission zip on the laptop (after the final run's output files are pulled).
#   bash code/final/make_package.sh final_v2 TEAMNAME
# Layout (as the problem statement asks):
#   <team>_submission.zip
#     output/matching_results.tsv, output/candidate_pairs.tsv
#     code/business_entity_resolution/{src/, README.md, requirements.txt}
#     Documentation_template.md
set -euo pipefail
RUN=$1
TEAM=${2:-team}
ROOT=$(pwd)
PKG=$ROOT/package
SRC=$PKG/code/business_entity_resolution/src
rm -rf "$SRC" "$PKG/output"
mkdir -p "$SRC" "$PKG/output"
for d in common blocking embeddings tfidf_fuzzy xgboost neural_reranker final eda baseline; do
  mkdir -p "$SRC/$d"
  cp "$ROOT/code/$d"/*.py "$SRC/$d/" 2>/dev/null || true
  cp "$ROOT/code/$d"/*.sh "$SRC/$d/" 2>/dev/null || true
done
cp "$ROOT/run_h100.py" "$SRC/"
cp "$ROOT/EXPERIMENT_LOG.md" "$SRC/"
cp "$ROOT/submissions/$RUN/matching_results.tsv" "$PKG/output/"
cp "$ROOT/submissions/$RUN/candidate_pairs.tsv" "$PKG/output/"
rm -f "$PKG/doc_sections_draft.md"
cd "$PKG"
py -3.10 - "$(cygpath -w "$ROOT/submissions/${TEAM}_submission.zip")" <<'PY'
import os, sys, zipfile
out = sys.argv[1]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=6) as z:
    for top in ("output", "code", "Documentation_template.md"):
        if os.path.isfile(top):
            z.write(top)
            continue
        for root, _, files in os.walk(top):
            for f in files:
                if "__pycache__" not in root:
                    z.write(os.path.join(root, f))
print("wrote", out, os.path.getsize(out) // 1_000_000, "MB")
PY
