#!/bin/bash
# Train a model for several seeds and evaluate it on the test set, with the settings of the paper:
# batch size 256, Adam with learning rate 1e-4, at most 5000 epochs, early stopping after 20 epochs without a new best
# validation RMSE, model.train() for training and model.eval() for validation and test (--bn_eval).
#
#   uv run scripts/train_and_test.sh --model Transformer_add_resid_eq2 --dataset anonymized_aligned --date 202610091700 --seeds "0 1 2 3 4"
#
# Options:
#   --model NAME      model name (main_completion.py; e.g. Transformer_add_resid_eq2, GNN_add_resid_eq2, TGAT_resid_eq2)
#   --dataset NAME    data set under <data_root>/dataset/ (anonymized, anonymized_aligned, ...)
#   --date LABEL      integer label of the experiment (directory <results_root>/weights/vel_1_acc_0_norm/<LABEL>/)
#   --seeds "S ..."   training seeds (default "0 1 2 3 4")
#   --epochs N        maximum number of epochs (default 5000)
#   --hs-alpha A      weight of the squared errors of players faster than 5.5 m/s (default 1 = unweighted)
#   --hs-oversample K draw training events with a player faster than 5.5 m/s K times as often (default 1 = off)
#   --test-only       evaluate the trained models of --date again without training
#   any other option is passed to main_completion.py (e.g. --data_root, --results_root, --numProcess)
set -euo pipefail

model=""; dataset=""; date=""; seeds="0 1 2 3 4"; epochs=5000; hs_alpha=1; hs_oversample=1; test_only=false; extra=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) model=$2; shift 2 ;;
        --dataset) dataset=$2; shift 2 ;;
        --date) date=$2; shift 2 ;;
        --seeds) seeds=$2; shift 2 ;;
        --epochs) epochs=$2; shift 2 ;;
        --hs-alpha) hs_alpha=$2; shift 2 ;;
        --hs-oversample) hs_oversample=$2; shift 2 ;;
        --test-only) test_only=true; shift ;;
        *) extra+=("$1"); shift ;;
    esac
done
if [[ -z "$model" || -z "$dataset" || -z "$date" ]]; then
    sed -n '2,20p' "$0"; exit 1
fi

PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."
for seed in $seeds; do
    args=(--data events --batchsize 256 --n_epoch "$epochs" --normalize --model "$model" --Beta 1.0 --start_lr 1e-4
          --do_prob 0.0 --date_preprocessed_data "$dataset" --hs_alpha "$hs_alpha" --hs_oversample "$hs_oversample"
          --bn_eval --seed "$seed" --date_experiment "$date" "${extra[@]}")
    if [[ "$test_only" != true ]]; then
        echo "=== train ${model} seed ${seed} (${dataset}) ==="
        $PYTHON main_completion.py "${args[@]}"
    fi
    echo "=== test ${model} seed ${seed} ==="
    $PYTHON main_completion.py "${args[@]}" --date_trained_model "$date" --TEST --n_sample 1
done
