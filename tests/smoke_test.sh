#!/bin/bash
# End-to-end check on synthetic data (no licensed data needed): synthetic data set -> anonymization ->
# association features -> window alignment -> training and testing of the proposed model and two baselines (2 epochs,
# 2 seeds) -> the analyses of the paper. Takes about 5 minutes on a CPU. The numbers it prints say nothing about the
# accuracy of the methods; the run only checks that every step works.
#
#   uv run tests/smoke_test.sh [WORK_DIR]        (default: a new temporary directory)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
W=${1:-$(mktemp -d)}
export VC_DATA_ROOT=$W/data VC_RESULTS_ROOT=$W/results
O=$W/out
mkdir -p "$O"
echo "working directory: $W"

step() { echo; echo "=== $* ==="; }

step "synthetic data set, anonymization, association features, alignment"
$PY examples/make_synthetic_dataset.py --dataset source --n_matches 4 2 2
$PY preprocess_data/anonymize_players.py --src source --dst anonymized > "$O/anonymize.log"
$PY preprocess_data/make_assoc_features.py --dataset anonymized
$PY preprocess_data/make_assoc_nonrecurrent.py --dataset anonymized
$PY preprocess_data/align_windows.py --dataset anonymized

step "training and testing (2 epochs, seeds 0 1)"
scripts/train_and_test.sh --model Transformer_add_resid_eq2 --dataset anonymized_aligned --date 1 --seeds "0 1" --epochs 2 \
    --numProcess 0 > "$O/train_proposed.log" 2>&1
for m in GNN_add_resid_eq2 TGAT_resid_eq2; do
    scripts/train_and_test.sh --model $m --dataset anonymized --date 2 --seeds "0 1" --epochs 2 --numProcess 0 > "$O/train_$m.log" 2>&1
done
grep -h "(L_rec2)" "$O"/train_*.log | sort -u

step "rule-based and position-difference baselines, PPCF/OBSO"
$PY analysis/interp_baseline.py --methods rule_based hungarian --ppcf --n_jobs 4 --cache_dir "$O/cache" --out "$O/interp.json" | grep -A3 "^method"
$PY analysis/rule_based_check.py --datasets source anonymized --n_sample 1 --n_jobs 4 --no_reference --out_dir "$O" | tail -1

step "speed bands, situations, PPCF/OBSO of the models"
$PY analysis/speed_metrics.py --date 2 --models GNN_add_resid_eq2 TGAT_resid_eq2 --seeds 0 1 --sample_dir sample_eval --out "$O/speed.json" | tail -1
$PY analysis/scenario_metrics.py --date 1 --models Transformer_add_resid_eq2 --seeds 0 1 --sample_dir sample_eval --n_jobs 4 \
    --cache_dir "$O/cache" --out "$O/scenario.json" --per_event_out "$O/scenario_per_event.npz" | tail -1

step "two-level bootstrap (matches x seeds)"
$PY analysis/pairwise_significance.py --date 2 --models Transformer_add_resid_eq2@1 GNN_add_resid_eq2 TGAT_resid_eq2 --seeds 0 1 \
    --bootstrap 200 --out "$O/pairwise.md" | tail -1
$PY analysis/pairwise_vs_baselines.py --model Transformer_add_resid_eq2 --date 1 --seeds 0 1 --model_per_event "$O/scenario_per_event.npz" \
    --baseline_per_event "$O/interp_per_event.npz" --bootstrap 200 --out "$O/pairwise_baselines.md" | tail -1

step "random forest with the same inputs"
$PY analysis/sklearn_resid_eq2_event.py --date 3 --models RF --variants resid_eq2 --seeds 0 | tail -1

step "alignment, equivariance, cost"
$PY analysis/alignment_accuracy.py --out "$O/alignment_accuracy.json" | tail -1
$PY analysis/aligned_pipeline_equivariance.py --run "$VC_RESULTS_ROOT/weights/vel_1_acc_0_norm/1/Transformer_add_resid_eq2_events_0_rectangular_256" \
    --n 50 --data "$VC_DATA_ROOT/dataset/anonymized/rectangular/recurrent" --out "$O/aligned_pipeline_equivariance.json" | tail -1
$PY analysis/equivariance_check.py --models GNN_add_resid_eq2 TGAT_resid_eq2 --date 2 --n_events 50 --out "$O/equivariance.json" | tail -1
$PY analysis/model_cost.py --date 1 --models Transformer_add_resid_eq2 --date_preprocessed_data anonymized_aligned --n_single 20 \
    --n_warmup 2 --out "$O/model_cost.json" | tail -2

echo
echo "SMOKE TEST PASSED (outputs in $O)"
