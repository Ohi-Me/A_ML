"""Build the final V6 submission ZIP on ANY computer (Windows / macOS / Linux, Python 3.8+, standard library only).

    <team>_submission.zip
    ├── output/matching_results.tsv          the file uploaded to the leaderboard
    ├── output/candidate_pairs.tsv           the blocking candidate set it was decoded from
    ├── code/business_entity_resolution/{src/, README.md, requirements.txt}
    └── Documentation_template.md            methodology, filled in from this run's result files

Run from the A_ML folder (the repository with code/, run_all.sh, results/ ...):

    python make_submission_v6.py --team "TEAM NAME" --members "Name 1, Name 2" --lb 0.988109

Options
  --outdir DIR     folder with the two V6 TSVs (default results/final/final_v6/output)
  --submitted F    the exact matching file uploaded to the leaderboard: must be byte-identical (checked)
  --choice F       results/v6/v6_choice.json if it lives elsewhere (numbers for the documentation)
  --plan F         results/v6/v6_plan.json if it lives elsewhere (final decoding settings)
  --bge yes|no     whether the second cross-encoder (bge-reranker-v2-m3) ran in V6, if its report is not in results/
Result: build/<team>_submission.zip. Missing optional result files only leave 'n/a' in the documentation.
"""
import argparse
import datetime
import filecmp
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))


def jl(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def find(name, *cands):
    for c in cands:
        if c and os.path.exists(os.path.join(ROOT, c)):
            return os.path.join(ROOT, c)
    for d, _, files in os.walk(ROOT):
        if name in files and "build" not in d.split(os.sep):
            return os.path.join(d, name)
    return None


def strip_v7(run_all):
    s = open(run_all, encoding="utf-8").read()
    if "# ================================================ V7 diagnostics" in s:
        a = s.index("# ================================================ V7 diagnostics")
        b = s.index("if [[ $LIST == 0 && $DRY == 0 ]]; then")
        s = s[:a] + s[b:]
    s = re.sub(r' \|\|\n\s*\( "\$STAGE" == "v7all"[^\n]*\n', " ]] || return 0\n", s)
    s = s.replace('''( "$STAGE" == "v6all" && "$block" =~ ^(v6diag|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ ) ]] || return 0''',
                  '''( "$STAGE" == "v6all" && "$block" =~ ^(v6diag|v6feat|v6s1|v6x|v6col|v6loco|v6test|v6final|v6pl|v6pick)$ ) ]] || return 0''')
    s = "\n".join(l for l in s.split("\n") if "--stage v7all" not in l and "v7final -> v7pick" not in l)
    s = s.replace("Choice: results/v7/v7_choice.json (V7) / results/v6/v6_choice.json (V6), first_choice_file",
                  "Choice: results/v6/v6_choice.json (first_choice_file)")
    return s


def final_args(plan):
    p = plan.get("plan", {})
    if not p:                                   # the settings of the submitted V6 run (v6_choice.json: caps on, EM off)
        p = {"caps": True, "em": False, "gamma_unseen": "same"}
    a = (["--caps"] if p.get("caps") else []) + (["--em"] if p.get("em") else [])
    return " ".join(a + ["--gamma_unseen", str(p.get("gamma_unseen", "same"))])


REPRODUCE = """#!/usr/bin/env bash
# Exactly the steps behind the submitted file (see ../README.md). Usage: bash reproduce_v6.sh /path/to/DATASET
# Resumable: every finished step is skipped when the command is run again.
set -euo pipefail
DATA="${1:?usage: bash reproduce_v6.sh /path/to/DATASET}"
cd "$(dirname "${BASH_SOURCE[0]}")"
bash run_all.sh --data "$DATA" --stage v2                                  # base: blocking, features, V2 models
for s in mlx_prep mlx_train mlx_prep_te mlx_score_te v6_dms; do            # XLM-R cross-encoder, density universe
  [ -e "${ER_CACHE:-data/cache_v2}/_done/$s" ] || bash run_all.sh --only "$s"; done
for b in v6feat v6s1 v6x v6col v6test; do bash run_all.sh --stage "$b"; done
export ER_ROOT="$PWD" ER_NORM=v2 ER_CACHE="${ER_CACHE:-data/cache_v2}" PYTHONPATH="$PWD/code"
python -u code/v6/final_v6.py --scores feats/test_scores_final_v6.parquet --tag xgb_v6_c2 @@FARGS@@ --out final_v6
bash code/final/validate.sh results/final/final_v6/output
echo "done: results/final/final_v6/output/{matching_results,candidate_pairs}.tsv"
"""


def count_candidates(path):
    n_pairs = n_s1 = 0
    with open(path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            n_s1 += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1]:
                n_pairs += parts[1].count(",") + 1
    return n_pairs, n_s1


def check_tsv(path, col):
    """light format check (the official validator needs the challenge data; it runs too if data/dataset exists)."""
    with open(path, encoding="utf-8") as fh:
        head = next(fh).rstrip("\n").split("\t")
        assert head == ["source1_entity_id", col], f"{path}: header {head}"
        seen, rows = set(), 0
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            assert len(parts) == 2, f"{path}: bad line {line[:80]}"
            assert parts[0] not in seen, f"{path}: duplicate S1 {parts[0]}"
            seen.add(parts[0])
            ids = [x for x in parts[1].split(",") if x]
            assert len(ids) == len(set(ids)), f"{path}: duplicate id in row of {parts[0]}"
            rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--members", default="[fill in]")
    ap.add_argument("--lb", default="[fill in]")
    ap.add_argument("--outdir", default=os.path.join("results", "final", "final_v6", "output"))
    ap.add_argument("--submitted", default="")
    ap.add_argument("--choice", default="")
    ap.add_argument("--plan", default="")
    ap.add_argument("--bge", default="auto", choices=["auto", "yes", "no"],
                    help="list BAAI/bge-reranker-v2-m3 as used (auto: detected from results/neural_reranker/ml_v6band or the V6 model report)")
    args = ap.parse_args()
    os.chdir(ROOT)
    match = os.path.join(args.outdir, "matching_results.tsv")
    cand = os.path.join(args.outdir, "candidate_pairs.tsv")
    for f in (match, cand):
        if not os.path.exists(f):
            sys.exit(f"missing {f}: pass --outdir with the folder that holds the V6 matching_results.tsv and candidate_pairs.tsv")
    if args.submitted:
        if not filecmp.cmp(args.submitted, match, shallow=False):
            sys.exit(f"ERROR: {args.submitted} differs from {match}")
        print("leaderboard file is byte-identical to", match)
    print("checking the two TSV files ...")
    n_rows_m = check_tsv(match, "matched_entity_ids")
    n_rows_c = check_tsv(cand, "candidate_entity_ids")
    assert n_rows_m == n_rows_c, (n_rows_m, n_rows_c)

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.team).strip("_") or "team"
    B = os.path.join(ROOT, "build", f"{safe}_submission")
    shutil.rmtree(B, ignore_errors=True)
    S = os.path.join(B, "code", "business_entity_resolution", "src")
    os.makedirs(os.path.join(B, "output"))
    print("copying outputs ...")
    shutil.copy2(match, os.path.join(B, "output", "matching_results.tsv"))
    shutil.copy2(cand, os.path.join(B, "output", "candidate_pairs.tsv"))
    print("copying source code ...")
    for d, dirs, files in os.walk("code"):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", "v7")]
        for f in files:
            if f.endswith((".py", ".sh")):
                dst = os.path.join(S, d)
                os.makedirs(dst, exist_ok=True)
                shutil.copy2(os.path.join(d, f), os.path.join(dst, f))
    os.makedirs(os.path.join(S, "docs"), exist_ok=True)
    for f in ("V6.md", "V6_RESEARCH.md", "EXPERIMENT_LOG.md"):
        if os.path.exists(f):
            shutil.copy2(f, os.path.join(S, "docs", f))
    val = os.path.join("Data", "student_resource", "utils", "validate_submission.py")
    if os.path.exists(val):
        os.makedirs(os.path.join(S, os.path.dirname(val)), exist_ok=True)
        shutil.copy2(val, os.path.join(S, val))
    shutil.copy2("requirements.txt", os.path.join(B, "code", "business_entity_resolution", "requirements.txt"))
    with open(os.path.join(S, "run_all.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(strip_v7("run_all.sh"))
    plan = jl(args.plan or find("v6_plan.json", os.path.join("results", "v6", "v6_plan.json")) or "")
    fargs = final_args(plan)
    with open(os.path.join(S, "reproduce_v6.sh"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(REPRODUCE.replace("@@FARGS@@", fargs))

    print("filling README and documentation ...")
    choice = jl(args.choice or find("v6_choice.json", os.path.join("results", "v6", "v6_choice.json")) or "")
    stats = jl(os.path.join(os.path.dirname(os.path.normpath(args.outdir)), "test_stats.json"))
    rep = jl(os.path.join("results", "xgboost", "xgb_v6_c2", "report.json"))
    cands = jl(os.path.join("results", "blocking", "candidates", "cands_c2_train.json"))
    bge = jl(os.path.join("results", "neural_reranker", "ml_v6band", "report_train.json"))
    models = ["`FacebookAI/xlm-roberta-base` (MIT, 278 M)"]
    used = bool(bge) or "bgexenc" in rep.get("args", {}).get("extra", "") or "bgexenc" in rep.get("features", [])
    if args.bge == "yes" or (args.bge == "auto" and used):
        models.append("`BAAI/bge-reranker-v2-m3` (Apache-2.0, 568 M)")
    mlist = " and ".join(models)
    n_pairs, n_s1 = count_candidates(cand)
    v6 = (choice.get("dms") or {}).get("V6") or {}
    bc = v6.get("by_country_f0", {})

    def f(x, d=5):
        return f"{x:.{d}f}" if isinstance(x, (int, float)) else "n/a"
    vf = min(v6["f0"], v6["f4"]) if v6 else None
    prec = sum(v["precision"] for v in bc.values()) / len(bc) if bc else None
    rec = sum(v["recall"] for v in bc.values()) / len(bc) if bc else None
    test_rows = ["| country | S1 | S1 with ≥ 1 match | matches | matches per S1 |", "|---|---|---|---|---|"]
    for c, v in (stats.get("by_country") or {}).items():
        test_rows.append(f"| {c} | {v['s1']:,} | {v.get('with_match', 0):,} | {v['matches']:,} | {v['matches_per_s1']:.3f} |")
    if len(test_rows) == 2:
        # from the matching file itself (S1 country not in it): overall only
        tot = 0
        with open(match, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                tot += (p[1].count(",") + 1) if len(p) > 1 and p[1] else 0
        test_rows.append(f"| all | {n_rows_m:,} | n/a | {tot:,} | {tot / max(n_rows_m, 1):.3f} |")
    val_rows = ["| country | F0.5 | precision | recall | FP | FN |", "|---|---|---|---|---|---|"]
    for c, v in bc.items():
        val_rows.append(f"| {c} | {v['f05']:.5f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['fp']:,} | {v['fn']:,} |")
    sub = {"{{TEAM}}": args.team, "{{MEMBERS}}": args.members, "{{DATE}}": datetime.date.today().isoformat(),
           "{{LB}}": args.lb, "{{MODEL_LIST}}": mlist, "{{FINAL_ARGS}}": f"`{fargs}`", "{{CAND_PAIRS}}": f"{n_pairs:,}",
           "{{CAND_PER_S1}}": f"{n_pairs / max(n_s1, 1):.1f}", "{{CAND_RECALL}}": f(cands.get("cand_recall_val"), 4),
           "{{CEILING}}": f((choice.get("ceiling_dms") or {}).get("f0")),
           "{{N_FEATURES}}": str(len(rep["features"])) if rep.get("features") else "about 130",
           "{{VAL_F05}}": f(vf), "{{VAL_P}}": f(prec, 4), "{{VAL_R}}": f(rec, 4),
           "{{TEST_TABLE}}": "\n".join(test_rows), "{{VAL_TABLE}}": "\n".join(val_rows)}
    for src, dst in ((os.path.join("submission_v6", "README.md"), os.path.join(B, "code", "business_entity_resolution", "README.md")),
                     (os.path.join("submission_v6", "Documentation_template.md"), os.path.join(B, "Documentation_template.md"))):
        s = open(src, encoding="utf-8").read()
        for k, v in sub.items():
            s = s.replace(k, v)
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
    print(json.dumps({k: v for k, v in sub.items() if "TABLE" not in k}, indent=1, ensure_ascii=False))

    if os.path.isdir(os.path.join("data", "dataset", "test")) and os.path.exists(os.path.join("data", "utils", "validate_submission.py")):
        print("official validator ...")
        subprocess.call([sys.executable, os.path.join("data", "utils", "validate_submission.py"), "--matching",
                         os.path.join(B, "output", "matching_results.tsv"), "--candidate", os.path.join(B, "output", "candidate_pairs.tsv"),
                         "--test-dir", os.path.join("data", "dataset", "test"), "--check-ids"])
    else:
        print("(official validator skipped: data/dataset/test not here; the files passed the format check above)")

    z = os.path.join(ROOT, "build", f"{safe}_submission.zip")
    if os.path.exists(z):
        os.remove(z)
    print("zipping ...")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for d, _, files in os.walk(B):
            for fn in files:
                p = os.path.join(d, fn)
                zf.write(p, os.path.relpath(p, B).replace(os.sep, "/"))
    h = hashlib.sha256(open(match, "rb").read()).hexdigest()
    print(f"\nready: {z}  ({os.path.getsize(z) / 2**20:.1f} MB)\nmatching_results.tsv sha256 {h}\n"
          f"check the team name / members in {os.path.join(B, 'Documentation_template.md')} before uploading")


if __name__ == "__main__":
    main()
