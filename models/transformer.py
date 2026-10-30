import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import event_config as cfg
from .mlp import MLP
from models.utils import loss_reconst, angular_error, axis_error

# Transformer for velocity completion (Sequence Model)
# Based on "Attention Is All You Need" (Vaswani et al., 2017)
# Processes sequence data with spatiotemporal self-attention

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE

# Frame rate of tracking data (Hz)
FRAME_RATE = 25.0
DT = 1.0 / FRAME_RATE


class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x, seq_dim=1):
        """
        x: (batch, seq_len, d_model) or (batch, seq_len, n_agents, d_model)
        """
        if x.dim() == 3:
            x = x + self.pe[:, : x.size(1), :]
        elif x.dim() == 4:
            # (batch, seq_len, n_agents, d_model)
            x = x + self.pe[:, : x.size(1), :].unsqueeze(2)
        return self.dropout(x)


class SpatialEncoding(nn.Module):
    """Learnable spatial encoding for agents."""

    def __init__(self, n_agents, d_model):
        super(SpatialEncoding, self).__init__()
        self.spatial_embed = nn.Embedding(n_agents, d_model)

    def forward(self, x):
        """
        x: (batch, n_agents, d_model) or (batch, seq_len, n_agents, d_model)
        """
        if x.dim() == 3:
            batch_size, n_agents, _ = x.size()
            positions = torch.arange(n_agents, device=x.device).unsqueeze(0).expand(batch_size, -1)
            return x + self.spatial_embed(positions)
        elif x.dim() == 4:
            batch_size, seq_len, n_agents, _ = x.size()
            positions = torch.arange(n_agents, device=x.device)
            spatial_enc = self.spatial_embed(positions)  # (n_agents, d_model)
            return x + spatial_enc.unsqueeze(0).unsqueeze(0)


class TypeEncoding(nn.Module):
    """Learnable embedding of the agent TYPE (ball / home player / away player) instead of the slot index.

    The slot index carries no information in the anonymized data (the players of a team are re-randomised at
    every event), so a per-slot embedding is noise; the type is the only thing the slots really distinguish and
    it is invariant to the relabelling inside a team (slot 0 = ball, 1..11 = home, 12..22 = away)."""

    def __init__(self, n_agents, d_model):
        super().__init__()
        self.type_embed = nn.Embedding(3, d_model)
        n_home = (n_agents - 1) // 2
        self.register_buffer("types", torch.tensor([0] + [1] * n_home + [2] * (n_agents - 1 - n_home)))

    def forward(self, x):
        e = self.type_embed(self.types)  # (n_agents, d_model)
        return x + (e.unsqueeze(0) if x.dim() == 3 else e.unsqueeze(0).unsqueeze(0))


class SpatioTemporalTransformerEncoder(nn.Module):
    """
    Transformer encoder with separate spatial and temporal attention.
    First applies attention across agents, then across time.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(SpatioTemporalTransformerEncoder, self).__init__()

        # Spatial attention (across agents)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Temporal attention (across time)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, n_agents, d_model)
        """
        batch_size, seq_len, n_agents, d_model = x.size()

        # Spatial attention: for each timestep, attend across agents
        x_spatial = x.reshape(batch_size * seq_len, n_agents, d_model)
        attn_out, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = self.spatial_norm(x_spatial + self.dropout(attn_out))
        x = x_spatial.reshape(batch_size, seq_len, n_agents, d_model)

        # Temporal attention: for each agent, attend across time
        x_temporal = x.permute(0, 2, 1, 3).reshape(batch_size * n_agents, seq_len, d_model)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = self.temporal_norm(x_temporal + self.dropout(attn_out))
        x = x_temporal.reshape(batch_size, n_agents, seq_len, d_model).permute(0, 2, 1, 3)

        # Feed-forward
        x = self.ffn_norm(x + self.ffn(x))

        return x


class Transformer(MLP):
    """
    Transformer model for player velocity completion.
    Processes sequence data with spatiotemporal self-attention.

    Input: (seq_len, batch, n_nodes, features)
    Output: velocity prediction for the final timestep
    """

    def __init__(self, params, parser=None):
        super(Transformer, self).__init__(params, parser=None)

        self.rnn_dim = params.get("rnn_dim", self.n_all_agents * 2)
        self.n_layers = params.get("n_layers", 1)

        self.n_heads = 4
        self.n_encoder_layers = 3
        # name tokens select the variants: assoc / resid (position-difference inputs), eq / eq2 (permutation-
        # equivariant head / fully equivariant), flat (non-equivariant head with a small hidden layer, to match the
        # number of parameters of the eq model), wide (d_model x 2, to match the parameters of the original model)
        name = params.get("model", "")
        toks = set(name.split("_"))
        self.wide = "wide" in toks
        self.flat = "flat" in toks
        self.d_model = self.n_hid * (8 if self.wide else 4)
        self.d_ff = self.d_model * 2

        self.n_input_add = 0
        self.eventtypes = params["eventtypes"]
        self.ballend = params["ballend"]
        # position-difference inputs (preprocess_data/make_assoc_features.py): displacement to the previous event
        # (players associated by position), the time gap and displacement / gap, for the target event only
        self.assoc = "assoc" in toks or "resid" in toks
        self.resid = "resid" in toks  # the network predicts the residual of the position-difference velocity
        # "_eq": one output head shared by all player slots (permutation-equivariant). The original head maps the
        # concatenation of all slots to all outputs, i.e. it uses a different weight for every slot index, which is
        # meaningless once the slots are re-randomised at every event (anonymized data)
        # "_eq2": fully permutation-equivariant inside a team: besides the shared head, the slot embedding is replaced by
        # a ball / home / away type embedding, and the event type and the ball end location are given to every
        # agent as the same input channels (instead of a learned feature per slot index: Linear(., 23))
        self.eq2 = "eq2" in toks
        self.eq = "eq" in toks or self.eq2
        self.n_assoc = 5
        if self.eventtypes:
            self.n_input_add += len(EVENTTYPE_CATEGORIES) if self.eq2 else 1
        if self.ballend:
            self.n_input_add += self.n_input if self.eq2 else 1
        if self.assoc:
            self.n_input_add += self.n_assoc

        # Input projection
        self.input_proj = nn.Linear(self.n_input + self.n_input_add, self.d_model)

        # Spatial encoding (for agents)
        self.spatial_encoding = (TypeEncoding if self.eq2 else SpatialEncoding)(self.n_all_agents, self.d_model)

        # Positional encoding (for temporal)
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=50, dropout=self.dropout_prob)

        # Spatiotemporal Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            SpatioTemporalTransformerEncoder(self.d_model, self.n_heads, self.d_ff, self.dropout_prob)
            for _ in range(self.n_encoder_layers)
        ])

        # Output projection (from final timestep)
        if self.eq:  # applied to the token of every player slot with the same weights
            self.output_proj = nn.Sequential(
                nn.Linear(self.d_model, self.d_model * 4),
                nn.GELU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(self.d_model * 4, self.n_output),
            )
        else:
            hidden = self.n_hid if self.flat else self.n_hid * self.n_all_agents
            self.output_proj = nn.Sequential(
                nn.Linear(self.d_model * self.n_all_agents, hidden),
                nn.GELU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(hidden, self.n_output * self.n_players),
            )

        if self.eventtypes and not self.eq2:
            self.fc_eventtype = nn.Linear(len(EVENTTYPE_CATEGORIES), self.n_all_agents)
        if self.ballend and not self.eq2:
            self.fc_ballend = nn.Linear(self.n_input, self.n_all_agents)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.resid:  # start from the position-difference velocity (the residual is zero)
            last = self.output_proj[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def feat_assoc(self, items, t, seq_len, batch_size):
        """(batch, n_agents, 5): association features at the target event, zeros at the earlier events."""
        if t != seq_len - 1:
            return torch.zeros(batch_size, self.n_all_agents, self.n_assoc, device=self.device)
        return items["assoc"][0].to(self.device)

    def add_difference(self, player_vel_pred, items):
        """Residual models: prediction = network output + velocity from the position difference [m/s]."""
        if not self.resid:
            return player_vel_pred
        return player_vel_pred + items["assoc"][0][:, 1:, 3:5].to(player_vel_pred.device) * 10.0

    def feat_ballend(self, ball_end_loc):
        batch_size = ball_end_loc.size(0)
        ball_end_loc = ball_end_loc.reshape(batch_size, -1).to(self.device).clone()
        if self.eq2:  # the same channels for every agent
            return ball_end_loc.unsqueeze(1).expand(batch_size, self.n_all_agents, -1)
        feat_ball_end = self.fc_ballend(ball_end_loc).reshape(
            batch_size, self.n_all_agents, -1
        )
        return feat_ball_end

    def feat_eventtype(self, event_type):
        batch_size = event_type.size(0)
        event_type = event_type.reshape(batch_size, -1).to(self.device).clone()
        if self.eq2:  # the same channels for every agent
            return event_type.unsqueeze(1).expand(batch_size, self.n_all_agents, -1)
        feat_event_type = self.fc_eventtype(event_type).reshape(
            batch_size, self.n_all_agents, -1
        )
        return feat_event_type

    def module(self, x):
        """
        x: (batch, seq_len, n_agents, n_input)
        """
        batch_size, seq_len, n_agents, _ = x.size()

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, n_agents, d_model)

        # Add spatial encoding
        x = self.spatial_encoding(x)

        # Add positional (temporal) encoding
        x = self.pos_encoding(x)

        # Apply spatiotemporal transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Take final timestep for output
        x_final = x[:, -1, :, :]  # (batch, n_agents, d_model)

        # Output projection
        if self.eq:  # the same head for every player token (slot 0 is the ball)
            return self.output_proj(x_final[:, 1:, :])
        x_out = x_final.reshape(batch_size, -1)
        x_out = self.output_proj(x_out).reshape(batch_size, self.n_players, 2)

        return x_out

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")

        # items["cies"]: (seq_len, batch, n_nodes, features)
        seq_len = items["cies"].size(0)
        batch_size = items["cies"].size(1)

        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            # Shape (1, batch_size) for compatibility with MODELS_SEQ concatenation
            batch_losses2["L_rec2"] = torch.zeros(1, batch_size).to(self.device)

        # Build sequence input: (batch, seq_len, n_agents, features)
        x_seq = []
        for t in range(seq_len):
            x_t = torch.cat(
                [
                    items["cies"][t, :, 0:1, :],
                    items["cies"][t, :, 2 : len(LOC_NODES), :],
                ],
                dim=1,
            ).to(self.device)
            x_t_tmp = x_t.clone()

            if self.ballend:
                feat_ball_end = self.feat_ballend(items["cies"][t, :, 1:2, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_ball_end], dim=2)
            if self.eventtypes:
                feat_event_type = self.feat_eventtype(items["eventtypes"][t, :, :, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)
            if self.assoc:
                x_t_tmp = torch.cat([x_t_tmp, self.feat_assoc(items, t, seq_len, batch_size)], dim=2)

            x_seq.append(x_t_tmp)

        # Stack to (batch, seq_len, n_agents, features)
        x_seq = torch.stack(x_seq, dim=1)

        # Forward pass
        v_t_pred = self.module(x_seq)

        # Get ground truth velocity for final timestep
        v_t = items["cies"][
            -1, :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
        ].to(self.device)

        # Loss (only for final timestep)
        player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
        player_vel_pred = self.add_difference(self.restore(v_t_pred, players_info), items)
        batch_losses["L_rec"] += loss_reconst(
            player_vel_pred, player_vel_GT, self.n_players, Sum=True
        )
        if acc == 0:
            batch_losses2["L_rec2"][0] += loss_reconst(
                player_vel_pred, player_vel_GT, self.n_players, Sum=False
            )

        return batch_losses, batch_losses2

    def sample(self, items, players_info, n_sample=1, TEST=False):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")

        seq_len = items["cies"].size(0)
        batch_size = items["cies"].size(1)

        if n_sample > 1:
            cies_n = [items["cies"].clone() for _ in range(n_sample)]
        else:
            cies_n = [items["cies"].clone()]

        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_rec"] = torch.zeros(n_sample).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(n_sample, batch_size).to(self.device)

        if TEST:
            data_test = {}
            data_test["v_pred"] = np.zeros((n_sample, batch_size, self.n_players, 2))
            data_test["v_gt"] = np.zeros((n_sample, batch_size, self.n_players, 2))
            data_test["v_rb"] = np.zeros((n_sample, batch_size, self.n_all_agents, 2))
            data_test["loc_gt"] = np.zeros((n_sample, batch_size, self.n_all_agents, 2))
            data_test["L_efobso"] = np.zeros((n_sample, batch_size))
            data_test["frames"] = np.zeros((n_sample, batch_size))
            data_test["L_angular"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_x"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_y"] = np.zeros((n_sample, batch_size))
            data_test["L_angular_rb"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_x_rb"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_y_rb"] = np.zeros((n_sample, batch_size))

        for i in range(n_sample):
            # Build sequence input
            x_seq = []
            for t in range(seq_len):
                x_t = torch.cat(
                    [
                        cies_n[i][t, :, 0:1, :],
                        cies_n[i][t, :, 2 : len(LOC_NODES), :],
                    ],
                    dim=1,
                ).to(self.device)
                x_t_tmp = x_t.clone()

                if self.ballend:
                    feat_ball_end = self.feat_ballend(cies_n[i][t, :, 1:2, :])
                    x_t_tmp = torch.cat([x_t_tmp, feat_ball_end], dim=2)
                if self.eventtypes:
                    feat_event_type = self.feat_eventtype(items["eventtypes"][t, :, :, :])
                    x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)
                if self.assoc:
                    x_t_tmp = torch.cat([x_t_tmp, self.feat_assoc(items, t, seq_len, batch_size)], dim=2)

                x_seq.append(x_t_tmp)

            x_seq = torch.stack(x_seq, dim=1)

            # Forward pass
            v_t_pred = self.module(x_seq)

            # Get data for final timestep
            t_final = seq_len - 1
            x_t_final = torch.cat(
                [
                    cies_n[i][t_final, :, 0:1, :],
                    cies_n[i][t_final, :, 2 : len(LOC_NODES), :],
                ],
                dim=1,
            ).to(self.device)
            v_t = cies_n[i][
                t_final, :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
            ].to(self.device)

            # Loss
            player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
            player_vel_pred = self.add_difference(self.restore(v_t_pred, players_info), items)
            batch_losses["L_rec"][i] += loss_reconst(
                player_vel_pred, player_vel_GT, self.n_players, Sum=True
            )
            if acc == 0:
                batch_losses2["L_rec2"][i] += loss_reconst(
                    player_vel_pred, player_vel_GT, self.n_players, Sum=False
                )

            # Position
            pos_GT = x_t_final.detach().cpu().numpy()
            max1 = cfg.FIELD_LENGTH / 2
            max2 = cfg.FIELD_WIDTH / 2
            pos_GT[:, :, 0] = pos_GT[:, :, 0] * max1
            pos_GT[:, :, 1] = pos_GT[:, :, 1] * max2

            if TEST:
                ball_vel_rb = np.expand_dims(v_t[:, 0, :].detach().cpu().numpy(), axis=1)
                player_vel_rb = cies_n[i][
                    t_final, :, len(LOC_NODES + VEL_NODES) :, :
                ].to(self.device)
                data_test["v_pred"][i] = player_vel_pred.detach().cpu().numpy()
                data_test["v_gt"][i] = player_vel_GT.detach().cpu().numpy()
                data_test["v_rb"][i] = np.concatenate(
                    (ball_vel_rb, player_vel_rb.detach().cpu().numpy()), axis=1
                )
                data_test["loc_gt"][i] = pos_GT
                data_test["L_efobso"][i] += (
                    loss_reconst(player_vel_rb, player_vel_GT, self.n_players, Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["L_angular"][i] += (
                    angular_error(player_vel_pred, player_vel_GT, self.n_players, Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["L_axis_x"][i] += (
                    axis_error(player_vel_pred, player_vel_GT, self.n_players, axis="x", Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["L_axis_y"][i] += (
                    axis_error(player_vel_pred, player_vel_GT, self.n_players, axis="y", Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["frames"][i] += (
                    items["frames"][:, t_final, :].detach().cpu().numpy().T[0]
                )
                data_test["L_angular_rb"][i] += (
                    angular_error(player_vel_rb, player_vel_GT, self.n_players, Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["L_axis_x_rb"][i] += (
                    axis_error(player_vel_rb, player_vel_GT, self.n_players, axis="x", Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )
                data_test["L_axis_y_rb"][i] += (
                    axis_error(player_vel_rb, player_vel_GT, self.n_players, axis="y", Sum=False)
                    .detach()
                    .cpu()
                    .numpy()
                )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
