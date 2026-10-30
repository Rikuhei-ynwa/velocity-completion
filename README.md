# Velocity completion for event-based player positional data in soccer

Code of the paper *Velocity Completion Task and Method for Event-based Player Positional Data in Soccer*
(Rikuhei Umemoto and Keisuke Fujii).

Event data record the positions of the ball and all players at passes, shots and other actions, but not the
players' instantaneous velocities. The task is to complete the velocities of all 22 players at an event from the
event-time positions, a short history of preceding events, their timing and context, without tracking trajectories
or persistent player identities. The proposed model is an aligned residual spatio-temporal Transformer: players are
associated across events by position-only Hungarian matching, the 10-event window is aligned backwards from the
target event, and a Transformer with parameters shared among players predicts a residual added to the
finite-difference velocity.

## What is released and what is not

* Released: the model implementations (proposed model and all learned baselines), training and evaluation, the
  steps of the method that operate on the data sets (anonymisation of the player slots, association features,
  window alignment), the rule-based baseline velocities, the metrics,
  the PPCF/OBSO evaluation, the statistical tests and the other analyses of the paper, a specification of the
  data format, and a synthetic data generator with an end-to-end test.
* Not released: the match data used in the paper. They were licensed from a data provider and cannot be
  redistributed, and neither can anything derived from them (data sets, predictions, trained weights). The
  preprocessing that turns the provider's data into the data sets (reading the provider's files, extracting the
  events, the attacking-third windows and the model-input arrays) is not released either.

**Consequently, the released artifact does not reproduce the numbers of the paper by itself.** It lets you inspect and
run the complete computation; the exact numbers require authorised access to the licensed data, converted to the
format of [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md). Results on the synthetic data only show that the code runs;
they say nothing about the accuracy of any method.

## Installation

Python 3.9 (tested with 3.9.19), PyTorch 2.4. A GPU is optional (the timing benchmark of the paper used a Tesla T4).
The environment is managed with [uv](https://docs.astral.sh/uv/); `pyproject.toml` and `uv.lock` pin the versions
used for the paper.

```bash
uv sync                          # creates .venv with Python 3.9 and the locked packages
uv sync --extra wandb            # optional: learning curves with main_completion.py --wandb
```

The PyTorch wheel installed on Linux is the CUDA 12.1 build from PyPI, which also runs on a CPU. Without uv,
`pip install -r requirements.txt` installs the same versions.

## Quick start on synthetic data

```bash
uv run tests/smoke_test.sh       # about 5 minutes on a CPU
```

It generates synthetic matches (`examples/make_synthetic_dataset.py`), builds the data sets, trains the proposed
model and two baselines for two epochs and runs the analyses. Step by step:

```bash
export VC_DATA_ROOT=$PWD/data VC_RESULTS_ROOT=$PWD/results
uv run python examples/make_synthetic_dataset.py --dataset source --n_matches 4 2 2
uv run python preprocess_data/anonymize_players.py --src source --dst anonymized
uv run python preprocess_data/make_assoc_features.py --dataset anonymized
uv run python preprocess_data/make_assoc_nonrecurrent.py --dataset anonymized
uv run python preprocess_data/align_windows.py --dataset anonymized
uv run scripts/train_and_test.sh --model Transformer_add_resid_eq2 --dataset anonymized_aligned --date 1 --seeds 0 --epochs 20
uv run python analysis/interp_baseline.py --methods rule_based hungarian
```

The commands in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) are written with `python`; run them with
`uv run` in front, or after `source .venv/bin/activate`.

## Using your own data

Build data sets in the format of [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) (positions of the ball and the 22 players
at every event, event types, ball end positions, frame numbers, and — for training and evaluation only — the
tracking-derived player velocities); `examples/make_synthetic_dataset.py` shows how the arrays are laid out. Then follow
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), which lists the command behind every result of the paper.

## Repository layout

| Path | Content |
|---|---|
| `main_completion.py` | training and testing of all neural models (and the original SVR / RF baselines) |
| `models/` | the proposed Transformer (`transformer.py`), the baselines, permutation-consistent layers (`equivariant.py`), association inputs and residual prediction (`assoc.py`) |
| `datasets/` | data loading, slot permutation, high-speed oversampling |
| `preprocess_data/` | anonymisation of the player slots, association features, window alignment, rule-based velocities |
| `analysis/` | baselines, metrics, PPCF/OBSO, bootstrap tests, cost, sensitivity analyses |
| `obso_settings.py`, `metrica/`, `DRSO/` | PPCF and OBSO |
| `examples/` | synthetic data sets in the released format |
| `scripts/train_and_test.sh` | training and testing with the settings of the paper |
| `tests/smoke_test.sh` | end-to-end test on synthetic data |

## License

MIT License (see [LICENSE](LICENSE)). Parts of the code are modified from other MIT-licensed projects; their notices
are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

See [CITATION.cff](CITATION.cff).
