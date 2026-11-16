#!/usr/bin/env bash
# Evaluate an intensity-augmented sweep and compare it with the baseline sweep.
#
#   bash scripts/evaluate_aug.sh [SWEEP] [BASELINE] [TEST_MRI_DIR]
#
# Defaults: output/aug_noN4, output/loss_sweep_final_22, dataset/Test_Data/MR.
# The test MRI folder must match the preprocessing the sweep was trained with
# (e.g. dataset/Test_Data/MR_N4 for configs/aug_full.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate

SWEEP=${1:-output/aug_noN4}
BASE=${2:-output/loss_sweep_final_22}
TEST_MR=${3:-dataset/Test_Data/MR}

n_logs=$(ls "$SWEEP"/logs/*.log | wc -l)
n_done=$(grep -l "Training complete" "$SWEEP"/logs/*.log | wc -l)
if [ "$n_done" -ne "$n_logs" ]; then
    echo "Only $n_done/$n_logs runs have finished in $SWEEP; aborting." >&2
    exit 1
fi

echo "== [1/4] Cross-validation evaluation"
rm -f "$SWEEP/eval_per_volume_best.csv"
python scripts/evaluate_sweep.py --sweep-root "$SWEEP" --use best

echo "== [2/4] Held-out test set, K=3 ensemble"
python scripts/ensemble_test.py --sweep-root "$SWEEP" --loss ssim \
    --test-mri-dir "$TEST_MR" --out-dir "$SWEEP/ensemble_test_ssim" --save-volumes

echo "== [3/4] Training curves"
python scripts/plot_training_curves.py --tb-dir "$SWEEP/tb" --loss ssim \
    --out "$SWEEP/training_curves.png"

echo "== [4/4] Comparison with $BASE"
python scripts/compare_runs.py --a "$BASE" --a-name baseline --b "$SWEEP" \
    --b-name "$(basename "$SWEEP")" --loss ssim --out "$SWEEP/comparison.md"
