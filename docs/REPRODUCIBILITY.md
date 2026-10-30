# Reproducing the analyses of the paper

The numbers of the paper were computed on synchronised event and tracking data of 55 matches of one league season (see the paper),
licensed from a data provider. These data cannot be redistributed, so the released code cannot regenerate the
reported values by itself. With data in the format of [DATA_SCHEMA.md](DATA_SCHEMA.md) the commands below run the
same computations; with the synthetic data of `examples/` they check that every step runs (`tests/smoke_test.sh`).

All commands are run from the repository root in the environment created by `uv sync` (prefix them with `uv run`, or
activate `.venv` first), with `VC_DATA_ROOT` (data sets) and `VC_RESULTS_ROOT` (trained runs) set. `--date` is an integer label of an experiment (the paper used the start time, e.g. 202610091700); the run directory of a model is
`$VC_RESULTS_ROOT/weights/vel_1_acc_0_norm/<date>/<model>_events_<seed>_rectangular_256/`, and the test predictions
are in `experiments/sample_eval/samples.p` of that directory. The labels that appear in the help texts of the analysis
scripts are those of the paper's runs.

## 1. Data sets

```bash
# <data_root>/dataset/source/: data sets in the format of DATA_SCHEMA.md (40 / 8 / 7 matches), built by the data owner
python preprocess_data/anonymize_players.py --src source --dst anonymized
python preprocess_data/make_assoc_features.py --dataset anonymized
python preprocess_data/make_assoc_nonrecurrent.py --dataset anonymized
python preprocess_data/align_windows.py --dataset anonymized                     # -> anonymized_aligned
```

## 2. Training and testing

Settings of the paper (in `scripts/train_and_test.sh`): batch size 256, Adam with learning rate 1e-4, at most 5000
epochs, early stopping after 20 epochs without improvement of the validation RMSE, five seeds (0–4), `model.eval()`
for validation and testing. The proposed model and the learned baselines of the main table:

```bash
# proposed: aligned residual spatio-temporal Transformer
scripts/train_and_test.sh --model Transformer_add_resid_eq2 --dataset anonymized_aligned --date 1

# learned baselines with association features, time gap, residual prediction and permutation-consistent layers
for m in MLP GNN GNN_add VAE VGNN VGNN_add CrossModalGating_add RNN VRNN GRNN GRNN_add GVRNN GVRNN_add TGAT; do
    scripts/train_and_test.sh --model ${m}_resid_eq2 --dataset anonymized --date 2
done

# support vector regression and random forest with the same inputs
python analysis/sklearn_resid_eq2_event.py --date 3 --models RF SVM --variants resid_eq2 --seeds 0 1 2 3 4
```

## 3. Where each result comes from

| Result in the paper | Runs | Command |
|---|---|---|
| Main RMSE table; rule-based and position-difference baselines | section 2 | `analysis/interp_baseline.py`, `analysis/speed_metrics.py` |
| Ablation of the proposed model | `Transformer_add`, `Transformer_add_resid`, `Transformer_add_resid_eq`, `Transformer_add_resid_eq2` on `anonymized`; `Transformer_add_resid_eq2` on `anonymized_aligned`; capacity controls `Transformer_add_flat`, `Transformer_add_eq2_wide` | `scripts/train_and_test.sh`, `analysis/pairwise_significance.py` |
| Two-level bootstrap over matches and seeds | any | `analysis/pairwise_significance.py`, `analysis/pairwise_vs_baselines.py` |
| PPCF / OBSO errors against tracking-derived velocities | proposed | `analysis/scenario_metrics.py --per_event_out ...`, `analysis/interp_baseline.py --ppcf` |
| Errors by speed band, angle error, speed bias | any | `analysis/speed_metrics.py` |
| Errors by tactical situation | any | `analysis/scenario_metrics.py` |
| Direction-only and speed-only counterfactuals | proposed | `analysis/direction_speed_decomposition.py` |
| High-speed weighted loss / oversampling | proposed with `--hs-alpha 2/5/10` or `--hs-oversample K` | `scripts/train_and_test.sh`, `analysis/speed_metrics.py` |
| Monte Carlo sensitivity of PPCF / OBSO to velocity noise | none | `analysis/mc_sensitivity.py --dataset source` |
| Sensitivity to direction errors | none | `analysis/angular_sensitivity.py` |
| Parameters, latency, throughput | any | `analysis/model_cost.py`; on an otherwise idle GPU `analysis/time_benchmark.py`; alignment cost `analysis/alignment_time.py` |
| Accuracy of the window alignment | none | `analysis/alignment_accuracy.py` |
| Permutation consistency of the full pipeline | proposed | `analysis/aligned_pipeline_equivariance.py`, `analysis/equivariance_check.py` |
| Spread of the predictions across seeds | any | `analysis/seed_uncertainty.py` |
| Rule-based baseline (checked against the values of the paper) | none | `analysis/rule_based_check.py` |

The PPCF/OBSO settings (pitch control parameters, EPV grid, ball transition model, 50 × 32 grid) are in
`obso_settings.py`. The figures of the paper are drawn from the outputs above; the plotting scripts are not part of
the release because they display licensed match data.

## 4. What the release does not contain

* the provider's raw data and every data set derived from them,
* the preprocessing that builds the data sets from the provider's data (reading the files, event extraction, the
  selection of the attacking-third events and windows, the model-input arrays),
* per-event, per-match or per-player predictions and errors of the paper's runs, and trained weights,
* the match ids of the split (the split rule is: sorted match ids, first 40 / next 8 / last 7).

Because the anonymisation is seeded with the provider's match ids, data sets built from other ids get other (equally
random) slot permutations.
