#!/usr/bin/env bash
# Reproduce every non-dosimetric result of the manuscript from a finished sweep.
#
#   bash scripts/reproduce_paper.sh [SWEEP_ROOT]      (default: output/loss_sweep_final_22)
#
# Writes figures, CSVs and report.md to <SWEEP_ROOT>/paper_repro/. The average CT
# is not rebuilt for the metrics: output/average_CT/average_ct.nii.gz (built with a
# manual 3D Slicer alignment step) is used as in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate

SWEEP=${1:-output/loss_sweep_final_22}
OUT=$SWEEP/paper_repro
TEST=dataset/Test_Data
AVG_CT=output/average_CT/average_ct.nii.gz
mkdir -p "$OUT"

done_runs=$(grep -l "Training complete" "$SWEEP"/logs/*.log 2>/dev/null | wc -l)
if [ "$done_runs" -ne 6 ]; then
    echo "Only $done_runs/6 training runs have finished in $SWEEP; aborting." >&2
    exit 1
fi
[ -f "$AVG_CT" ] || { echo "Missing $AVG_CT" >&2; exit 1; }

echo "== [1/6] Cross-validation evaluation (Table 1)"
rm -f "$SWEEP/eval_per_volume_best.csv"
python scripts/evaluate_sweep.py --sweep-root "$SWEEP" --use best
python scripts/compare_losses.py --per-volume-csv "$SWEEP/eval_per_volume_best.csv"

echo "== [2/6] Held-out test set, K=3 ensembles (Section 3.2, Table S1)"
for loss in ssim l1; do
    python scripts/ensemble_test.py --sweep-root "$SWEEP" --loss "$loss" \
        --test-mri-dir "$TEST/MR" --out-dir "$SWEEP/ensemble_test_$loss" --save-volumes
done

echo "== [3/6] Bone DSC of sCT and aligned avCT, avCT MAE (Section 3.2, Table S1)"
python scripts/test_dice_table.py --sweep-root "$SWEEP" --test-root "$TEST" \
    --avg-ct "$AVG_CT" --out "$OUT/dice_test_cohort"

echo "== [4/6] Bone-overlay figure (Fig. 1)"
python scripts/make_combined_dice_figure.py --sweep-root "$SWEEP" --test-root "$TEST" \
    --subject MZEI00031_MR_T1 --out "$OUT" --avg-ct-out "$OUT/average_ct_rebuilt.nii.gz"

echo "== [5/6] Training curves (Suppl. Fig. S3)"
python scripts/plot_training_curves.py --tb-dir "$SWEEP/tb" --loss ssim \
    --out "$OUT/training_curves.png"

echo "== [6/6] Report"
python scripts/reproduce_report.py --sweep-root "$SWEEP" \
    --dice-csv "$OUT/dice_test_cohort.csv" --out "$OUT/report.md"

echo
echo "Done. Results in $OUT"
