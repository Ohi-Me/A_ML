#!/bin/bash
# official validator on a submission folder: bash code/final/validate.sh results/final/<tag>/output
OUT=$1
python -u data/utils/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir data/dataset/test --check-ids
