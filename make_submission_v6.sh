#!/usr/bin/env bash
# =====================================================================================================================
#  Build the final submission ZIP for V6 in the structure the organisers require:
#
#   <team>_submission.zip
#   ├── output/matching_results.tsv          (the file uploaded to the leaderboard)
#   ├── output/candidate_pairs.tsv           (the blocking candidate set it was decoded from)
#   ├── code/business_entity_resolution/{src/, README.md, requirements.txt}
#   └── Documentation_template.md            (filled in from this run's result files)
#
#  Run it on the machine where V6 ran, from the repository folder:
#    bash make_submission_v6.sh --team "TEAM_NAME" --members "Name 1, Name 2" --lb 0.988109 \
#         [--submitted /path/to/the/file/you/uploaded.tsv] [--outdir results/final/final_v6/output]
#  --submitted: the exact matching file uploaded to the leaderboard; the script checks it is byte-identical.
# =====================================================================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
TEAM=""; MEMBERS="[fill in]"; LB="[fill in]"; SUBMITTED=""; OUTDIR="results/final/final_v6/output"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --team) TEAM="$2"; shift 2;;
    --members) MEMBERS="$2"; shift 2;;
    --lb) LB="$2"; shift 2;;
    --submitted) SUBMITTED="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    *) echo "unknown argument $1"; exit 2;;
  esac
done
[[ -n "$TEAM" ]] || { echo "--team is required"; exit 2; }
SAFE="$(echo "$TEAM" | tr ' /' '__')"
for f in "$OUTDIR/matching_results.tsv" "$OUTDIR/candidate_pairs.tsv" results/v6/v6_plan.json; do
  [[ -f "$f" ]] || { echo "missing $f (run the V6 pipeline first)"; exit 3; }
done
if [[ -n "$SUBMITTED" ]]; then
  cmp -s "$SUBMITTED" "$OUTDIR/matching_results.tsv" || { echo "ERROR: $SUBMITTED differs from $OUTDIR/matching_results.tsv"; exit 4; }
  echo "leaderboard file is byte-identical to $OUTDIR/matching_results.tsv"
fi

B="build/${SAFE}_submission"
rm -rf "$B"; mkdir -p "$B/output" "$B/code/business_entity_resolution/src"
S="$B/code/business_entity_resolution/src"

echo "== output files"
cp "$OUTDIR/matching_results.tsv" "$OUTDIR/candidate_pairs.tsv" "$B/output/"

echo "== source code"
mkdir -p "$S/code" "$S/docs" "$S/Data/student_resource/utils"
( cd code && find . -type f \( -name '*.py' -o -name '*.sh' \) -not -path './v7/*' -not -path '*/__pycache__/*' -print0 ) |
  while IFS= read -r -d '' f; do mkdir -p "$S/code/$(dirname "$f")"; cp "code/$f" "$S/code/$f"; done
cp Data/student_resource/utils/validate_submission.py "$S/Data/student_resource/utils/"
cp V6.md V6_RESEARCH.md EXPERIMENT_LOG.md "$S/docs/"
cp requirements.txt "$B/code/business_entity_resolution/requirements.txt"
# the runner without the V7 blocks (V7 is not part of this submission)
python3 - "$S/run_all.sh" <<'PY'
import sys
s = open("run_all.sh").read()
a = s.index("# ================================================ V7 diagnostics")
b = s.index("if [[ $LIST == 0 && $DRY == 0 ]]; then")
s = s[:a] + s[b:]
s = s.replace('''       ( "$STAGE" == "v7all" && "$block" =~ ^(v7diag|v7cands|v7feat|v7s1|v7xenc|v7col|v7test|v7final|v7pick)$ ) ]] || return 0''',
              ''' ]] || return 0''').replace('''( "$STAGE" == "v6all" && "$block" =~ ^(v6diag|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ ) ||\n''',
                                            '''( "$STAGE" == "v6all" && "$block" =~ ^(v6diag|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ )''')
s = s.replace("""#                 V7 on a V6 cache: --stage v7all (v7diag -> v7cands -> v7feat -> v7s1 -> v7xenc -> v7col -> v7test ->
#                 v7final -> v7pick, see V7.md)
""", "")
s = s.replace("Choice: results/v7/v7_choice.json (V7) / results/v6/v6_choice.json (V6), first_choice_file",
              "Choice: results/v6/v6_choice.json (first_choice_file)")
assert "v7" not in s.lower(), [l for l in s.splitlines() if "v7" in l.lower()][:3]
open(sys.argv[1], "w").write(s)
PY
bash -n "$S/run_all.sh"
chmod +x "$S/run_all.sh"

FARGS="$(PYTHONPATH=code python3 code/v6/choose_v6.py final_args)"
cat > "$S/reproduce_v6.sh" <<EOF
#!/usr/bin/env bash
# Exactly the steps behind the submitted file (see ../README.md). Usage: bash reproduce_v6.sh /path/to/DATASET
# Resumable: every finished step is skipped when the command is run again.
set -euo pipefail
DATA="\${1:?usage: bash reproduce_v6.sh /path/to/DATASET}"
cd "\$(dirname "\${BASH_SOURCE[0]}")"
bash run_all.sh --data "\$DATA" --stage v2                                  # base: blocking, features, V2 models
for s in mlx_prep mlx_train mlx_prep_te mlx_score_te v6_dms; do            # XLM-R cross-encoder, density universe
  [ -e "\${ER_CACHE:-data/cache_v2}/_done/\$s" ] || bash run_all.sh --only "\$s"; done
for b in v6feat v6s1 v6x v6col v6test; do bash run_all.sh --stage "\$b"; done
export ER_ROOT="\$PWD" ER_NORM=v2 ER_CACHE="\${ER_CACHE:-data/cache_v2}" PYTHONPATH="\$PWD/code"
python -u code/v6/final_v6.py --scores feats/test_scores_final_v6.parquet $FARGS --out final_v6
bash code/final/validate.sh results/final/final_v6/output
echo "done: results/final/final_v6/output/{matching_results,candidate_pairs}.tsv"
EOF
chmod +x "$S/reproduce_v6.sh"

echo "== README and documentation"
python3 - "$B" "$TEAM" "$MEMBERS" "$LB" "$FARGS" "$OUTDIR" <<'PY'
import datetime, glob, json, os, sys
B, team, members, lb, fargs, outdir = sys.argv[1:7]
def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}
choice, stats = jl("results/v6/v6_choice.json"), jl("results/final/final_v6/test_stats.json")
rep = jl("results/xgboost/xgb_v6_c2/report.json")
cands = jl("results/blocking/candidates/cands_c2_train.json")
bge = jl("results/neural_reranker/ml_v6band/report_train.json")
models = ["`FacebookAI/xlm-roberta-base` (MIT, 278 M)"]
if bge and os.path.exists("data/cache_v2/xenc/ml_v6band_model_A.pt"):
    models.append(f"`{bge.get('model', 'BAAI/bge-reranker-v2-m3')}` (Apache-2.0, 568 M)")
extras = rep.get("args", {}).get("extra", "")
if "bgexenc" not in extras and len(models) > 1:
    models = models[:1]
mlist = " and ".join(models)
# candidate pairs in the submitted file
n_pairs, n_s1 = 0, 0
with open(os.path.join(outdir, "candidate_pairs.tsv"), encoding="utf-8") as fh:
    next(fh)
    for line in fh:
        n_s1 += 1
        ids = line.rstrip("\n").split("\t")[1] if "\t" in line else ""
        n_pairs += len([x for x in ids.split(",") if x])
v6 = (choice.get("dms") or {}).get("V6") or {}
bc = v6.get("by_country_f0", {})
tp = {c: v for c, v in bc.items()}
def f(x, d=5):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"
val = min(v6["f0"], v6["f4"]) if v6 else None
prec = sum(v["precision"] for v in tp.values()) / len(tp) if tp else None
rec = sum(v["recall"] for v in tp.values()) / len(tp) if tp else None
test_rows = ["| country | S1 | S1 with ≥ 1 match | matches | matches per S1 |", "|---|---|---|---|---|"]
for c, v in (stats.get("by_country") or {}).items():
    test_rows.append(f"| {c} | {v['s1']:,} | {v.get('with_match', 0):,} | {v['matches']:,} | {v['matches_per_s1']:.3f} |")
val_rows = ["| country | F0.5 | precision | recall | FP | FN |", "|---|---|---|---|---|---|"]
for c, v in bc.items():
    val_rows.append(f"| {c} | {v['f05']:.5f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['fp']:,} | {v['fn']:,} |")
sub = {"{{TEAM}}": team, "{{MEMBERS}}": members, "{{DATE}}": datetime.date.today().isoformat(), "{{LB}}": lb,
       "{{MODEL_LIST}}": mlist, "{{FINAL_ARGS}}": f"`{fargs}`", "{{CAND_PAIRS}}": f"{n_pairs:,}",
       "{{CAND_PER_S1}}": f"{n_pairs / max(n_s1, 1):.1f}",
       "{{CAND_RECALL}}": f(cands.get("cand_recall_val"), 4), "{{CEILING}}": f((choice.get("ceiling_dms") or {}).get("f0")),
       "{{N_FEATURES}}": str(len(rep.get("features", []))) if rep else "n/a",
       "{{VAL_F05}}": f(val), "{{VAL_P}}": f(prec, 4), "{{VAL_R}}": f(rec, 4),
       "{{TEST_TABLE}}": "\n".join(test_rows), "{{VAL_TABLE}}": "\n".join(val_rows)}
for src, dst in (("submission_v6/README.md", os.path.join(B, "code/business_entity_resolution/README.md")),
                 ("submission_v6/Documentation_template.md", os.path.join(B, "Documentation_template.md"))):
    s = open(src, encoding="utf-8").read()
    for k, v in sub.items():
        s = s.replace(k, v)
    left = [k for k in sub if k in s]
    assert not left, left
    open(dst, "w", encoding="utf-8").write(s)
print(json.dumps({k: v for k, v in sub.items() if "TABLE" not in k}, indent=1, ensure_ascii=False))
PY

echo "== official validator on output/"
if [[ -d data/dataset/test ]]; then
  bash code/final/validate.sh "$B/output"
else
  echo "WARNING: data/dataset/test not found - validator skipped"
fi

echo "== zip"
ZIP="build/${SAFE}_submission.zip"
rm -f "$ZIP"
( cd "$B" && zip -q -r -9 "../${SAFE}_submission.zip" output code Documentation_template.md )
ls -la "$ZIP"
unzip -l "$ZIP" | awk 'NR<=3 || /output\/|README|requirements|Documentation|reproduce_v6|run_all/' | head -20
sha256sum "$B/output/matching_results.tsv"
echo "ready: $ZIP  (check the team name / members in $B/Documentation_template.md before uploading)"
