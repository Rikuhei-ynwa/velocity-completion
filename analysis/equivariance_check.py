"""Are the models permutation-equivariant in the player slots? (anonymized data, benchmark fairness)

In the anonymized data the slots of the players of a team are re-randomised at every event, so the slot index
carries no information. A model suited to that data should be permutation-equivariant: relabelling the player
slots of the input (the same relabelling for the whole window, within each team, ball fixed) must relabel the
predicted velocities the same way. A model whose output layer has a different weight for every slot index
(for example a flatten + linear head) is not.

Two checks on the first --n_events test events:
  structure  untrained (randomly initialised) model, deterministic (the reparameterisation noise of the
             generative models is set to zero): the largest difference between the prediction for the relabelled
             input and the relabelled prediction for the original input [m/s]. ~1e-6: equivariant by construction
  trained    the same with the trained weights of the official runs: how much the trained prediction still depends
             on the slot labelling [mean |difference| per player, m/s]

For the Transformer family the learned slot embedding (SpatialEncoding) is a second source of non-equivariance;
--zero_slot_embedding repeats the structural check with that embedding set to zero, which isolates the output head.

    python analysis/equivariance_check.py [--zero_slot_embedding]
"""

import argparse, json, os, pickle, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
from datasets import EventDataset, MODELS_SEQ
from models import load_model

N_AGENTS = 23
HOME, AWAY = np.arange(1, 12), np.arange(12, 23)
LOC0, VEL0, FAKE0 = 0, 24, 47  # layout of cies (preprocess_data/make_dataset.py)
recv, send = np.where(np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS))
pair2edge = -np.ones((N_AGENTS, N_AGENTS), dtype=np.int64)
pair2edge[recv, send] = np.arange(len(recv))

OFFICIAL = "202610021638"
TRAINED = {  # model -> date of the official / new runs (seed 0)
    **{m: OFFICIAL for m in "MLP GNN GNN_add VAE VGNN VGNN_add RNN GRNN GRNN_add VRNN GVRNN GVRNN_add GAT CrossModalGating GATRNN Transformer TGAT".split()},
    "Transformer_add": "202610030117",
    "Transformer_add_assoc": "202610031836", "Transformer_add_resid": "202610031836",
    "Transformer_add_eq": "202610031948", "Transformer_add_assoc_eq": "202610031948", "Transformer_add_resid_eq": "202610031948",
}
EQ_BASES = ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add",
            "GAT", "CrossModalGating", "GATRNN", "TGAT"]
_SEQ_RESID = {"RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add"}
TRAINED.update({
    "Transformer_add_eq2": "202610041317", "Transformer_add_assoc_eq2": "202610041317", "Transformer_add_resid_eq2": "202610041317",
    **{m + "_eq": "202610041900" for m in ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN",
                                           "GVRNN_add", "GAT", "CrossModalGating", "GATRNN", "TGAT"]},
    **{m + "_eqw": "202610051115" for m in ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN",
                                            "GVRNN_add", "GAT", "CrossModalGating", "GATRNN", "TGAT"]},
    **{m + "_resid_eq2": ("202610071600" if m in _SEQ_RESID else "202610061720")
       for m in ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add", "GAT",
                 "CrossModalGating", "GATRNN", "TGAT"]},
})
CONFIG_FROM = {  # models without trained weights yet: the configuration (params.p) of the sibling run
    "Transformer_add_eq2": "Transformer_add_eq", "Transformer_add_assoc_eq2": "Transformer_add_assoc_eq",
    "Transformer_add_resid_eq2": "Transformer_add_resid_eq",
    **{m + "_eq": m for m in EQ_BASES},
    **{m + "_eqw": m for m in EQ_BASES},
    **{m + "_resid_eq2": m for m in EQ_BASES},
}


def node_index(P):
    pl = P[..., 1:]
    lead = P.shape[:-1]
    return np.concatenate([np.broadcast_to([LOC0, LOC0 + 1], lead + (2,)), LOC0 + 1 + pl,
                           np.broadcast_to([VEL0], lead + (1,)), VEL0 + pl, FAKE0 - 1 + pl], axis=-1)


def relabel(items, P, is_seq):
    """Apply the slot relabelling P (B, T, 23) (new slot k holds old agent P[k]) to a batch in dataset layout."""
    Pb = P if is_seq else P[:, 0]
    out = dict(items)
    idx = torch.as_tensor(node_index(Pb))
    out["cies"] = torch.gather(items["cies"], -2, idx.unsqueeze(-1).expand_as(items["cies"]))
    Pt = torch.as_tensor(Pb)
    if "node_type" in items:
        nt = items["node_type"]
        out["node_type"] = torch.gather(nt, -2, Pt.unsqueeze(-1).expand(*Pt.shape, nt.shape[-1]))
    if "edge_type" in items:
        et = items["edge_type"]
        eidx = torch.as_tensor(pair2edge[Pb[..., recv], Pb[..., send]])
        out["edge_type"] = torch.gather(et, -2, eidx.unsqueeze(-1).expand_as(et))
    if "assoc" in items:  # (B, 1, 23, 5) sequence layout or (B, 23, 5) single event, agent axis = -2
        a = items["assoc"]
        Pa = torch.as_tensor(P[:, -1:]) if a.dim() == 4 else torch.as_tensor(Pb)  # the features of the target (last) event
        out["assoc"] = torch.gather(a, -2, Pa.unsqueeze(-1).expand_as(a))
    return out


def independent_perm(n, T, rng):
    """An independent random within-team relabelling for every event of the window: the symmetry of the anonymized data
    (preprocess_data/anonymize_players.py relabels every event independently)."""
    P = np.tile(np.arange(N_AGENTS), (n, T, 1))
    for team in (HOME, AWAY):
        P[:, :, team] = team[np.argsort(rng.random((n, T, len(team))), axis=-1)]
    return P


def static_perm(n, T, rng):
    """One random within-team relabelling per event, the same for every event of the window."""
    P = np.tile(np.arange(N_AGENTS), (n, 1))
    for team in (HOME, AWAY):
        P[:, team] = team[np.argsort(rng.random((n, len(team))), axis=-1)]
    return np.repeat(P[:, None, :], T, axis=1)


@torch.no_grad()
def predict(model, items, is_seq, players_info):
    it = {k: (v if k == "frames" else v.permute(1, 0, 2, 3)) for k, v in items.items()} if is_seq else items
    saved = torch.Tensor.normal_
    torch.Tensor.normal_ = lambda self, *a, **k: self.zero_()  # the noise of the generative models: zero (deterministic)
    try:
        _, _, data = model.sample(it, players_info, n_sample=1, TEST=True)
    finally:
        torch.Tensor.normal_ = saved
    return np.asarray(data["v_pred"][0])  # (B, 22, 2)


def randomise(model, seed=0):
    """Fresh random weights (Xavier) for every matrix: some models start with a zero output layer (the output
    would be constant and trivially equivariant)."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            std = (2.0 / (p.shape[0] + p[0].numel())) ** 0.5
            p.data = torch.randn(p.shape, generator=g) * std


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--models", nargs="+", default=list(TRAINED))
    cli.add_argument("--n_events", type=int, default=128)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--zero_slot_embedding", action="store_true")
    cli.add_argument("--perm", default="static", choices=["static", "independent"],
                     help="static: the same relabelling for every event of the window (window-level equivariance); independent: an "
                          "independent relabelling for every event, as in the anonymized data (the symmetry the data actually has)")
    cli.add_argument("--fail_above", type=float, default=None,
                     help="exit with an error if the structural max |difference| of any listed model is above this value")
    cli.add_argument("--zero_slot_features", action="store_true",
                     help="Transformer family: also zero fc_eventtype / fc_ballend (Linear(., 23): a different learned "
                          "feature for every slot index)")
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--out", default="analysis/results/equivariance_check.json")
    cli.add_argument("--date", default=None,
                     help="experiment label of the trained runs (seed 0) of all --models; default: the labels of the paper's runs")
    cli.add_argument("--dataset", default="anonymized", help="data set of the test events")
    opt = cli.parse_args()
    if opt.date is not None:
        TRAINED.update({m: opt.date for m in opt.models})

    rng = np.random.default_rng(opt.seed)
    results = {}
    print(f"{'model':<26}{'seq':>4}{'structure: max |diff|':>23}{'(mean |v|)':>11}{'equivariant':>13}{'trained: mean |diff|':>22}{'mean |v|':>10}")
    for name in opt.models:
        src = name if name in TRAINED else CONFIG_FROM.get(name, name)  # no trained weights: borrow the configuration of the sibling run
        wdir = os.path.join(opt.weights_dir, TRAINED[src], f"{src}_events_0_rectangular_256")
        params = dict(pickle.load(open(os.path.join(wdir, "params.p"), "rb")))
        params["model"] = name
        is_seq = name in MODELS_SEQ
        ds = EventDataset(argparse.Namespace(data_path=opt.data_path, batchsize=opt.n_events, model=name, n_all_agents=N_AGENTS,
                                               date_preprocessed_data=opt.dataset), train=-1)
        items = next(iter(DataLoader(ds, batch_size=opt.n_events, shuffle=False)))
        n = items["cies"].shape[0]
        T = items["cies"].shape[1] if is_seq else 1
        P = (independent_perm if opt.perm == "independent" else static_perm)(n, T, rng)
        items_p = relabel(items, P, is_seq)
        order = torch.as_tensor(P[:, -1, 1:] - 1).numpy()  # old player index of every new slot at the target (last) event

        def diff(model):
            v0, v1 = predict(model, items, is_seq, ds.players_vel_info), predict(model, items_p, is_seq, ds.players_vel_info)
            v0p = np.take_along_axis(v0, order[..., None], axis=1)  # the prediction for the original input, relabelled
            d = np.abs(v1 - v0p)
            return float(d.max()), float(d.mean()), float(np.linalg.norm(v0, axis=-1).mean())

        torch.manual_seed(0)
        model = load_model(name, params)
        randomise(model)
        model.eval()
        if opt.zero_slot_embedding and hasattr(model, "spatial_encoding"):
            model.spatial_encoding.spatial_embed.weight.data.zero_()
        if opt.zero_slot_features:
            for part in ("fc_eventtype", "fc_ballend"):
                if hasattr(model, part):
                    for q in getattr(model, part).parameters():
                        q.data.zero_()
        s_max, s_mean, s_v = diff(model)

        if name not in TRAINED:  # no trained weights
            t_mean, vm = float("nan"), s_v
        else:
            model = load_model(name, params)  # trained weights
            model.load_state_dict(torch.load(os.path.join(wdir, "state_dict_best.pth"), map_location="cpu", weights_only=True))
            model.eval()
            _, t_mean, vm = diff(model)
        eq = s_max < 1e-4
        results[name] = {"sequence_model": is_seq, "structure_max_abs_diff": s_max, "structure_mean_abs_diff": s_mean,
                         "structure_mean_speed": s_v, "equivariant_by_structure": eq,
                         "trained_mean_abs_diff": t_mean, "mean_speed": vm}
        print(f"{name:<26}{'yes' if is_seq else 'no':>4}{s_max:>23.2e}{s_v:>11.2f}{'YES' if eq else 'no':>13}{t_mean:>22.4f}{vm:>10.2f}", flush=True)

    json.dump({"n_events": opt.n_events, "perm": opt.perm, "zero_slot_embedding": opt.zero_slot_embedding,
               "zero_slot_features": opt.zero_slot_features, "results": results},
              open(opt.out, "w"), indent=1)
    print(f"Saved: {opt.out}")
    if opt.fail_above is not None:
        bad = [m for m, r in results.items() if r["structure_max_abs_diff"] > opt.fail_above]
        if bad:
            sys.exit(f"not equivariant (structural max |difference| > {opt.fail_above}): {' '.join(bad)}")


if __name__ == "__main__":
    main()
