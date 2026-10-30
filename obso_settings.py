"""Settings of the PPCF/OBSO computation (the definitions of the space-evaluation pipeline of the paper).

PPCF: potential pitch control field (metrica/Metrica_PitchControl.py, default model parameters),
OBSO: off-ball scoring opportunity (DRSO/obso_player.py) with the EPV grid and the ball transition model
of DRSO/EPV_grid.csv and DRSO/Transition_gauss.csv, on a 50 x 32 grid of a 105 m x 68 m pitch.
"""

import argparse, os

import numpy as np
import pandas as pd

import metrica.Metrica_PitchControl as mpc
import metrica.Metrica_EPV as mepv

HERE = os.path.dirname(os.path.abspath(__file__))


def settings(data_root, dataset, seq_len=10, datatype="recurrent"):
    """Namespace with the PPCF/OBSO parameters and the path of the test metadata (<split>.pkl)."""
    args = argparse.Namespace()
    args.seq_len = seq_len
    args.datatype = datatype
    args.cies = "rectangular"
    args.data_test_path = os.path.join(data_root, "dataset", dataset, args.cies, datatype, "test.pkl")
    args.field_dimen = (105.0, 68.0)
    args.params = mpc.default_model_params()
    EPV = mepv.load_EPV_grid(os.path.join(HERE, "DRSO", "EPV_grid.csv"))
    args.EPV = EPV / np.max(EPV)
    Trans = np.array(pd.read_csv(os.path.join(HERE, "DRSO", "Transition_gauss.csv"), header=None))
    args.Trans = Trans / np.max(Trans)
    args.n_grid_cells_x = 50
    args.n_grid_cells_y = int(args.n_grid_cells_x * args.field_dimen[1] / args.field_dimen[0])
    args.optimal = False
    args.offsides = True
    return args
