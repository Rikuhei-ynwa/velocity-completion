from .mlp import MLP
from .gnn import GNN
from .vae import VAE
from .vgnn import VGNN
from .rnn import RNN
from .grnn import GRNN
from .vrnn import VRNN
from .gvrnn import GVRNN
from .svm import SVM
from .random_forest import RandomForest
from .gat import GAT
from .transformer import Transformer
from .gatrnn import GATRNN
from .crossmodal_gating import CrossModalGating
from .tgat import TGAT
from .equivariant import make_equivariant

# models whose slot-specific layers are replaced by permutation-equivariant ones in "<name>_eq"
# (models/equivariant.py); the model code itself is not changed
EQ_BASES = [
    "MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add",
    "GAT", "CrossModalGating", "CrossModalGating_add", "GATRNN", "TGAT",
]

"""
MODELS_NON_RNN = [
    'VGNN_add', 'GNN_add', 'VGNN', 'VAE', 'GNN', 'MLP',
    'GAT', 'Transformer', 'Transformer_add', 'CrossModalGating', 'TGAT',
    ]
MODELS_RNN = [
    'GVRNN_add', 'GVRNN', 'VRNN', 'GRNN', 'RNN', 'GATRNN',
    ]
"""


# width multipliers found by the search below for n_all_agents = 23, n_hid = 8 (the configuration of every run);
# other configurations are searched. Parameters of "<name>_eqw" / "<name>": 1.00 +- 0.03 (analysis/eq_parameter_breakdown.py)
_EQW_MULT = {
    ("MLP", 23, 8): 14, ("GNN", 23, 8): 9, ("GNN_add", 23, 8): 9, ("VAE", 23, 8): 33, ("VGNN", 23, 8): 22,
    ("VGNN_add", 23, 8): 22, ("RNN", 23, 8): 34, ("GRNN", 23, 8): 23, ("GRNN_add", 23, 8): 23, ("VRNN", 23, 8): 37,
    ("GVRNN", 23, 8): 25, ("GVRNN_add", 23, 8): 25, ("GAT", 23, 8): 11, ("CrossModalGating", 23, 8): 9,
    ("GATRNN", 23, 8): 24, ("TGAT", 23, 8): 23,
}


def _n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _eqw_build(base, params, parser, mult):
    p = dict(params, model=base)
    if "rnn_dim" in p and "RNN" in base:  # per-agent hidden size of the GRU: 2 * mult (the original: 2)
        p["rnn_dim"] = params["n_all_agents"] * 2 * mult
    return make_equivariant(load_model(base, p, parser), mult)


def _eqw_mult(base, params, parser):
    """The width multiplier whose equivariant model has the number of parameters closest to the original model."""
    key = (base, params["n_all_agents"], params["n_hid"])
    if key not in _EQW_MULT:
        target = _n_params(load_model(base, dict(params, model=base), parser))
        best, best_gap = 1, None
        for mult in range(1, 129):
            gap = abs(_n_params(_eqw_build(base, params, parser, mult)) - target)
            if best_gap is None or gap < best_gap:
                best, best_gap = mult, gap
            elif _n_params(_eqw_build(base, params, parser, mult)) > target:
                break
        _EQW_MULT[key] = best
    return _EQW_MULT[key]


def load_model(model_name, params, parser=None):
    if model_name.endswith("_resid_eq2") and model_name[: -len("_resid_eq2")] in EQ_BASES:
        # the model with the association inputs, the residual prediction and the equivariant layers (models/assoc.py)
        from models.assoc import N_ASSOC, make_assoc
        base = model_name[: -len("_resid_eq2")]
        p = dict(params, model=base, n_input=params["n_input"] + N_ASSOC)
        return make_assoc(make_equivariant(load_model(base, p, parser)))
    if model_name.endswith("_eq") and model_name[: -len("_eq")] in EQ_BASES:
        base = model_name[: -len("_eq")]
        return make_equivariant(load_model(base, dict(params, model=base), parser))
    if model_name.endswith("_eqw") and model_name[: -len("_eqw")] in EQ_BASES:
        # the equivariant model with as many parameters as the original one (parameter-matched control)
        base = model_name[: -len("_eqw")]
        return _eqw_build(base, params, parser, _eqw_mult(base, params, parser))
    if model_name == "MLP":
        return MLP(params, parser)
    elif model_name == "GNN" or model_name == "GNN_add":
        return GNN(params, parser)
    elif model_name == "VAE":
        return VAE(params, parser)
    elif model_name == "VGNN" or model_name == "VGNN_add":
        return VGNN(params, parser)
    elif model_name == "RNN":
        return RNN(params, parser)
    elif model_name == "GRNN" or model_name == "GRNN_add":
        return GRNN(params, parser)
    elif model_name == "VRNN":
        return VRNN(params, parser)
    elif model_name == "GVRNN" or model_name == "GVRNN_add":
        return GVRNN(params, parser)
    elif model_name == "SVM":
        return SVM(params, parser)
    elif model_name == "RF":
        return RandomForest(params, parser)
    elif model_name == "GAT":
        return GAT(params, parser)
    elif model_name.startswith("Transformer"):
        return Transformer(params, parser)
    elif model_name == "GATRNN":
        return GATRNN(params, parser)
    elif model_name == "CrossModalGating" or model_name == "CrossModalGating_add":
        return CrossModalGating(params, parser)
    elif model_name == "TGAT":
        return TGAT(params, parser)
    else:
        raise NotImplementedError
