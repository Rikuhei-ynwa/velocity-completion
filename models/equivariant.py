"""Permutation-equivariant drop-in replacements for the slot-specific parts of the models.

In the anonymized data the players of a team are re-randomised at every event, so the slot index of a player
carries no information. The models of the paper were built for jersey-ordered data: they flatten the 23 agent
slots (ball + 22 players) into one vector and apply layers whose weights differ for every slot index. Such a
layer has to learn the same rule once per slot, from data in which every slot holds a random player. A
permutation-equivariant layer applies one shared rule to every agent (Deep Sets, Zaheer et al. 2017).

Every module here keeps the INPUT AND OUTPUT SHAPES of the layer it replaces (the tensors are flattened
agent-major, (batch, agents * channels)), so the surrounding code of the models does not change:

  EqBatchNorm     replaces BatchNorm1d(agents * C): one set of C statistics shared by all agents
  EqReadout       replaces Linear(agents * C_in -> agents_out * K): the same small MLP for every agent, on its own
                  features, its agent type (ball / home / away) and the pooled features of the ball, the home and
                  the away players (the pooling gives the agents the information of the others)
  EqGRU           replaces GRU(agents * C -> agents * H): one GRU cell shared by all agents (same pooled context)
  BroadcastLinear replaces Linear(n -> agents) used to give the event type / ball end to every slot with a
                  different learned weight: one value for all agents

The ball is slot 0, the home players are slots 1..11 and the away players 12..22; the relabelling that is
allowed (and that the layers are equivariant to) is any permutation inside a team.

make_equivariant(model) replaces these layers in a constructed model; the replaced layers are recognised by
their shapes (see the rules in _replace).
"""

import torch
import torch.nn as nn


def agent_types(n_agents):
    n_home = (n_agents - 1) // 2
    return torch.tensor([0] + [1] * n_home + [2] * (n_agents - 1 - n_home))


class _Context(nn.Module):
    """[own features, agent type one-hot, ball features, mean of the home players, mean of the away players]."""

    def __init__(self, n_agents):
        super().__init__()
        self.A = n_agents
        self.n_home = (n_agents - 1) // 2
        self.register_buffer("types", nn.functional.one_hot(agent_types(n_agents), 3).float())

    def forward(self, h):  # (B, A, C) -> (B, A, 4C + 3)
        B, A, C = h.shape
        ball = h[:, :1].expand(B, A, C)
        home = h[:, 1:1 + self.n_home].mean(1, keepdim=True).expand(B, A, C)
        away = h[:, 1 + self.n_home:].mean(1, keepdim=True).expand(B, A, C)
        return torch.cat([h, self.types.unsqueeze(0).expand(B, A, 3).to(h.dtype), ball, home, away], dim=-1)


class EqBatchNorm(nn.Module):
    def __init__(self, n_agents, c):
        super().__init__()
        self.A, self.C = n_agents, c
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):  # (B, A*C)
        B = x.shape[0]
        return self.bn(x.reshape(B * self.A, self.C)).reshape(B, self.A * self.C)


class EqReadout(nn.Module):
    def __init__(self, n_agents, c_in, k_out, players_only, hidden=None):
        super().__init__()
        self.A, self.C, self.K, self.players_only = n_agents, c_in, k_out, players_only
        hidden = hidden or max(16, 2 * c_in)
        self.ctx = _Context(n_agents)
        self.net = nn.Sequential(nn.Linear(4 * c_in + 3, hidden), nn.ELU(), nn.Linear(hidden, k_out))

    def forward(self, x):  # (B, A*C_in) -> (B, A_out*K)
        B = x.shape[0]
        out = self.net(self.ctx(x.reshape(B, self.A, self.C)))  # (B, A, K)
        if self.players_only:
            out = out[:, 1:]
        return out.reshape(B, -1)


class EqGRU(nn.Module):
    """Same call as nn.GRU(input (S, B, A*C), h0 (L, B, A*H)) -> (output (S, B, A*H), h_n (L, B, A*H))."""

    def __init__(self, n_agents, c_in, hid, n_layers):
        super().__init__()
        self.A, self.C, self.H, self.L = n_agents, c_in, hid, n_layers
        self.ctx = _Context(n_agents)
        self.cells = nn.ModuleList([nn.GRUCell(4 * (c_in if l == 0 else hid) + 3, hid) for l in range(n_layers)])

    def forward(self, inp, h0):
        S, B, _ = inp.shape
        h = [h0[l].reshape(B * self.A, self.H) for l in range(self.L)]
        outs = []
        for s in range(S):
            x = inp[s].reshape(B, self.A, self.C)
            for l in range(self.L):
                h[l] = self.cells[l](self.ctx(x).reshape(B * self.A, -1), h[l])
                x = h[l].reshape(B, self.A, self.H)
            outs.append(x.reshape(B, self.A * self.H))
        return torch.stack(outs), torch.stack([hl.reshape(B, self.A * self.H) for hl in h])


class BroadcastLinear(nn.Module):
    """Linear(n_in -> n_agents) that gives the SAME value to every agent (instead of one weight per slot)."""

    def __init__(self, n_in, n_agents):
        super().__init__()
        self.A = n_agents
        self.lin = nn.Linear(n_in, 1)

    def forward(self, x):
        return self.lin(x).expand(-1, self.A)


def _replace(model, mult=1):
    """Replace the slot-specific layers of a constructed model. Returns the list of replacements.
    mult scales the hidden width of the shared readouts (the number of parameters is a free design choice of the
    equivariant layers: they do not depend on the number of agents)."""
    A = model.n_all_agents
    P = A - 1
    done = []
    for pname, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            full = f"{pname}.{cname}".lstrip(".")
            new = None
            if isinstance(child, nn.BatchNorm1d) and child.num_features % A == 0 and child.num_features > A:
                new = EqBatchNorm(A, child.num_features // A)
            elif isinstance(child, nn.Linear):
                i, o = child.in_features, child.out_features
                if cname in ("fc_eventtype", "fc_ballend") and o == A:
                    new = BroadcastLinear(i, A)
                elif i % A == 0 and i // A >= 2 and (o % A == 0 or o % P == 0):
                    new = EqReadout(A, i // A, o // A if o % A == 0 else o // P, players_only=o % A != 0,
                                    hidden=max(16, 2 * (i // A)) * mult)
            elif isinstance(child, nn.GRU) and child.input_size % A == 0 and child.hidden_size % A == 0:
                new = EqGRU(A, child.input_size // A, child.hidden_size // A, child.num_layers)
            if new is not None:
                setattr(parent, cname, new)
                done.append((full, type(child).__name__, type(new).__name__))
    return done


def make_equivariant(model, mult=1):
    done = _replace(model, mult)
    model.equivariant_layers = done
    return model
