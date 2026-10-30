"""Off-ball scoring opportunity (OBSO) of one event.

Modified from https://github.com/keisuke198619/C-OBSO (MIT License, Copyright (c) 2022 Keisuke Fujii; see
THIRD_PARTY_NOTICES.md). Only the function used by this repository is kept.
"""

import numpy as np

import SPADL_config as spc


def calc_obso(
    PPCF: float,
    Transition: np.array,
    Score: np.array,
    ball_start_pos: np.array,
    n_grid_cells_x: int,
    n_grid_cells_y: int,
    attack_direction=0
):
    # calculate obso in single frame
    # PPCF, Score : 50 * 32
    # Transition : 100 * 64
    Transition = np.array((Transition))
    Score = np.array((Score))
    ball_grid_x = int(
        (ball_start_pos[0]
         + (spc.FIELD_LENGTH / 2))
         // (spc.FIELD_LENGTH / n_grid_cells_x)
         )
    ball_grid_y = int(
        (ball_start_pos[1]
         + (spc.FIELD_WIDTH / 2))
        // (spc.FIELD_WIDTH / n_grid_cells_y)
        )

    # When out of the pitch
    if ball_grid_x < 0:
        ball_grid_x = 0
    elif ball_grid_x > n_grid_cells_x - 1:
        ball_grid_x = n_grid_cells_x - 1
    if ball_grid_y < 0:
        ball_grid_y = 0
    elif ball_grid_y > n_grid_cells_y - 1:
        ball_grid_y = n_grid_cells_y - 1

    Transition = Transition[
        (n_grid_cells_y - 1) - ball_grid_y : (n_grid_cells_y * 2 - 1) - ball_grid_y,
        (n_grid_cells_x - 1) - ball_grid_x : (n_grid_cells_x * 2 - 1) - ball_grid_x
        ]

    if attack_direction < 0:
        Score = np.fliplr(Score)
    elif attack_direction > 0:
        Score = Score
    else:
        print("input attack direction is 1 or -1")

    obso = PPCF * Transition * Score

    return obso, Transition
