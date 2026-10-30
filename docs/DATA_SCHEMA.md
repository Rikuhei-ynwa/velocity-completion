# Data format

The models and analyses read data sets under `<data_root>/dataset/<name>/rectangular/`, described below. Building
these data sets from the raw data of a provider (reading the files, extracting the events and the tracking-derived
velocities, selecting the target events) is the data owner's preprocessing and is not part of this repository; the
rule-based velocities can be computed with `preprocess_data/rule_based_velocity.py`. `examples/make_synthetic_dataset.py` writes synthetic data sets in this format.

Units: positions in metres, velocities in m/s. Coordinates: origin at the centre spot, x along the length of the
pitch (−52.5 … 52.5), y along the width (−34 … 34), on a 105 m × 68 m pitch. Times are frame numbers of the
tracking data at 25 Hz (0 is reserved for padding).

Matches are split into training / validation / test (40 / 8 / 7 in the paper). The match ids are the keys of the
`.pkl` files; `preprocess_data/anonymize_players.py` converts them to integers, so use zero-padded integers
(`"000"`, `"001"`, …).

## Data sets


`<data_root>/dataset/<name>/rectangular/{recurrent,non_recurrent}/`:

* `recurrent`: windows of the 10 events ending at a target event (target = last event). Events before the start of
  the match or of the second half are padding (`frames` = 0, positions −4 after scaling, type `non_action`).
* `non_recurrent`: the target events alone. Its test set has the same target events as the recurrent test set.
* Target events (paper): events whose ball start and end positions both lie in the attacking third (beyond 17.5 m
  from the centre line towards the goal attacked by the team of the event) and whose ball velocity is known.
* Ground-truth velocities: instantaneous player velocities at the event frame from the tracking data, used only as
  training targets and for evaluation. Missing players: position (−210, −136) before scaling, velocity 0.

## `<split>.npz` (`split` = `training`, `validation`, `test`)

N = number of samples, T = 10 (recurrent only; the arrays of `non_recurrent` have no T axis).

| Key | Shape | Content |
|---|---|---|
| `cies` | (N, T, 69, 2) float64 | rows 0–23: positions of ball start, ball end, Home 1–11, Away 1–11, divided by (52.5, 34); rows 24–46: velocities of the ball and the 22 players, standardised with `vel_stats`; rows 47–68: velocities of the rule-based baseline [m/s] (`preprocess_data/rule_based_velocity.py`), not scaled |
| `vel_stats` | (2, 4) | mean_x, std_x, mean_y, std_y of the ball (row 0) and of all players (row 1) velocities of this split |
| `node_type` | (N, T, 23, 5) | per agent (ball, 22 players): is ball, is player, attack direction, on the ball, goalkeeper |
| `edge_type` | (N, T, 506, 2) | per directed edge of the complete graph on 23 agents: same team, ball–actor relation |
| `eventtypes` | (N, T, 18) | one-hot event type (order of `event_config.EVENT_TYPE`) |
| `frames` | (N, T, 1) int | frame numbers (0 = padding). Not a model input except for the time gaps of TGAT; also used to match samples |

## `<split>.pkl`

`dict[match_id, dict]` with the samples of every match in the order of the `.npz` (the match of every test sample is
needed for the match-level bootstrap). Keys per match: `frames`, `types` (str), `locs` (positions in metres),
`vels`, `vel_fakes`, `direcs`, `onballs`, `gks` (each with shape (n, T, …) as in the npz), and the row indices
`idx_filtered` (and `idx_series_filtered`): indices of the target event (and of the window events) in the event
sequence of the match.

## Anonymised and aligned data sets

* `anonymized` (`preprocess_data/anonymize_players.py`): the 11 player slots of every team are permuted
  independently at every event (seeded by the seed 0, the match id and the frame), so the slot order carries no
  identity across events. Adds `<split>_permutations.npy` (N, T, 23): the original slot of every new slot. The
  permutation maps are used only by diagnostics (alignment accuracy), never by a model.
  Note: the paper's data set was relabelled with the provider's match ids; with other match ids the permutations
  (not the method) differ.
* `<split>_assoc.npz` (`preprocess_data/make_assoc_features.py`, and `make_assoc_nonrecurrent.py` for the single-event
  data): key `assoc` (N, 1, 23, 5), the association features of the target event — displacement since the previous
  event with the players associated by position (Hungarian matching inside each team) / 10 m, the time gap / 10 s,
  and the finite-difference velocity / 10 m/s (players capped at 12 m/s, the ball at 40 m/s). Zero if there is no
  previous event.
* `anonymized_aligned` (`preprocess_data/align_windows.py`, recurrent only): the earlier events of every window are
  re-ordered backwards from the target event by position-only Hungarian matching, so that a slot follows the same
  (estimated) player through the window. The target event and the association features are unchanged.
