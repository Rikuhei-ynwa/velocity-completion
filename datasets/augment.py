"""Training-time utilities for the symmetry experiments (analysis of the equivariant models).

permute_slots   random relabelling of the player slots inside each team, applied to a training batch: the same
                relabelling for the whole window of a sample, ball fixed (data augmentation by the group of the
                anonymized data; the alternative to building the symmetry into the architecture)
SubsetWithAttrs a random fraction of the training windows (learning curves); attributes of the dataset
                (players_vel_info, ...) are passed through
"""

import numpy as np
import torch

N_AGENTS = 23
LOC0, VEL0, FAKE0 = 0, 24, 47  # layout of cies (preprocess_data/make_dataset.py)


def permute_slots(items, is_seq):
    """items in dataset layout: cies (B, [T,] 69, 2); assoc (B, 1, 23, 5) is permuted along the agent axis too.
    Graph inputs (node_type, edge_type) are not supported (the models of these experiments have none)."""
    if "node_type" in items or "edge_type" in items:
        raise NotImplementedError("permute_slots: graph inputs are not supported")
    cies = items["cies"]
    B = cies.shape[0]
    P = torch.arange(N_AGENTS).repeat(B, 1)
    P[:, 1:12] = 1 + torch.argsort(torch.rand(B, 11), dim=-1)
    P[:, 12:] = 12 + torch.argsort(torch.rand(B, 11), dim=-1)
    pl = P[:, 1:]
    zero = torch.zeros(B, 1, dtype=torch.long)
    idx = torch.cat([zero + LOC0, zero + LOC0 + 1, LOC0 + 1 + pl, zero + VEL0, VEL0 + pl, FAKE0 - 1 + pl], dim=1)  # (B, 69)
    index = idx.view(B, 1, -1, 1).expand_as(cies) if is_seq else idx.view(B, -1, 1).expand_as(cies)
    out = dict(items)
    out["cies"] = torch.gather(cies, -2, index)
    if "assoc" in items:
        a = items["assoc"]
        index = P.view(B, 1, N_AGENTS, 1) if a.dim() == 4 else P.view(B, N_AGENTS, 1)  # sequence (B, 1, 23, 5) / single event (B, 23, 5)
        out["assoc"] = torch.gather(a, -2, index.expand_as(a))
    return out


class SubsetWithAttrs(torch.utils.data.Subset):
    def __init__(self, dataset, fraction, seed):
        n = len(dataset)
        rng = np.random.default_rng(seed)
        super().__init__(dataset, np.sort(rng.choice(n, max(1, int(round(n * fraction))), replace=False)).tolist())

    def __getattr__(self, name):  # players_vel_info, ball_vel_info, ...
        if name in ("dataset", "indices"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def fast_event_weights(dataset, oversample, threshold=5.5):
    """Sampling weights for the high-speed data augmentation (reviewers R2-6, R4-2): an event in which at least one player is
    faster than `threshold` m/s at the target event (the last event of the window) is drawn `oversample` times as often as the
    others. Used with torch.utils.data.WeightedRandomSampler (the number of samples per epoch stays the same)."""
    cies = dataset.cies
    last = cies[:, -1] if cies.dim() == 4 else cies  # (N, 69, 2): the target event
    v = last[:, VEL0 + 1 : VEL0 + N_AGENTS, :].float()  # players (the ball is VEL0)
    info = dataset.players_vel_info
    vx = v[..., 0] * float(info[1]) + float(info[0])
    vy = v[..., 1] * float(info[3]) + float(info[2])
    fast = (torch.sqrt(vx ** 2 + vy ** 2) > threshold).any(dim=1)
    w = torch.ones(len(fast), dtype=torch.double)
    w[fast] = float(oversample)
    return w, float(fast.float().mean())
