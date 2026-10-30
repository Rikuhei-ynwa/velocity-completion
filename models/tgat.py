import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import event_config as cfg
from .mlp import MLP
from models.utils import add_difference, loss_reconst, angular_error, axis_error

# Temporal Graph Attention Network for velocity completion (Sequence Model)
# Based on "Inductive Representation Learning on Temporal Graphs" (Xu et al., 2020)
# Uses time-aware attention with temporal kernel functions

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE

# Frame rate of tracking data (Hz)
FRAME_RATE = 25.0
DT = 1.0 / FRAME_RATE  # Default time between frames in seconds


class TimeEncode(nn.Module):
    """
    Time encoding using Bochner's theorem and random Fourier features.
    Maps time deltas to high-dimensional features.
    """

    def __init__(self, d_model):
        super(TimeEncode, self).__init__()
        self.d_model = d_model
        # Learnable frequency basis
        self.w = nn.Linear(1, d_model)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, d_model))).float().reshape(d_model, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(d_model).float())

    def forward(self, t):
        """
        t: time values of any shape
        Returns: time encoding with d_model features added as last dimension
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        else:
            t = t.unsqueeze(-1)
        output = torch.cos(self.w(t))
        return output


class TemporalGraphAttentionLayer(nn.Module):
    """
    Temporal Graph Attention Layer for sequence data.
    Incorporates both spatial attention (across agents) and temporal attention (across time).
    """

    def __init__(self, in_features, out_features, time_dim, n_heads=4, dropout=0.1):
        super(TemporalGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        self.time_dim = time_dim

        assert out_features % n_heads == 0, "out_features must be divisible by n_heads"

        # Spatial attention (across agents)
        self.W_q_spatial = nn.Linear(in_features, out_features)
        self.W_k_spatial = nn.Linear(in_features, out_features)
        self.W_v_spatial = nn.Linear(in_features, out_features)

        # Temporal attention (across time) with time encoding
        self.W_q_temporal = nn.Linear(in_features, out_features)
        self.W_k_temporal = nn.Linear(in_features, out_features)
        self.W_v_temporal = nn.Linear(in_features, out_features)
        self.W_t = nn.Linear(time_dim, out_features)  # Time encoding projection

        # Output projections
        self.W_o_spatial = nn.Linear(out_features, out_features)
        self.W_o_temporal = nn.Linear(out_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.spatial_norm = nn.LayerNorm(out_features)
        self.temporal_norm = nn.LayerNorm(out_features)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q_spatial, self.W_k_spatial, self.W_v_spatial,
                       self.W_q_temporal, self.W_k_temporal, self.W_v_temporal,
                       self.W_t, self.W_o_spatial, self.W_o_temporal]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, time_enc):
        """
        x: (batch, seq_len, n_agents, in_features)
        time_enc: (batch, seq_len, time_dim)

        Returns: (batch, seq_len, n_agents, out_features)
        """
        batch_size, seq_len, n_agents, _ = x.size()

        # ===== Spatial Attention (across agents for each timestep) =====
        x_spatial = x.reshape(batch_size * seq_len, n_agents, -1)

        Q_s = self.W_q_spatial(x_spatial)
        K_s = self.W_k_spatial(x_spatial)
        V_s = self.W_v_spatial(x_spatial)

        # Multi-head attention
        Q_s = Q_s.view(-1, n_agents, self.n_heads, self.head_dim).transpose(1, 2)
        K_s = K_s.view(-1, n_agents, self.n_heads, self.head_dim).transpose(1, 2)
        V_s = V_s.view(-1, n_agents, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores_s = torch.matmul(Q_s, K_s.transpose(-2, -1)) / scale
        attn_s = F.softmax(scores_s, dim=-1)
        attn_s = self.dropout(attn_s)

        out_s = torch.matmul(attn_s, V_s)
        out_s = out_s.transpose(1, 2).contiguous().view(-1, n_agents, self.out_features)
        out_s = self.W_o_spatial(out_s)
        out_s = self.dropout(out_s)

        x_spatial = self.spatial_norm(x_spatial[:, :, :self.out_features] + out_s)
        x = x_spatial.reshape(batch_size, seq_len, n_agents, -1)

        # ===== Temporal Attention (across time for each agent) =====
        x_temporal = x.permute(0, 2, 1, 3).reshape(batch_size * n_agents, seq_len, -1)

        Q_t = self.W_q_temporal(x_temporal)
        K_t = self.W_k_temporal(x_temporal)
        V_t = self.W_v_temporal(x_temporal)

        # Add time encoding to keys
        time_enc_expanded = time_enc.unsqueeze(1).expand(-1, n_agents, -1, -1)
        time_enc_flat = time_enc_expanded.reshape(batch_size * n_agents, seq_len, -1)
        T = self.W_t(time_enc_flat)
        K_t = K_t + T

        # Multi-head attention
        Q_t = Q_t.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K_t = K_t.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_t = V_t.view(-1, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores_t = torch.matmul(Q_t, K_t.transpose(-2, -1)) / scale
        attn_t = F.softmax(scores_t, dim=-1)
        attn_t = self.dropout(attn_t)

        out_t = torch.matmul(attn_t, V_t)
        out_t = out_t.transpose(1, 2).contiguous().view(-1, seq_len, self.out_features)
        out_t = self.W_o_temporal(out_t)
        out_t = self.dropout(out_t)

        x_temporal = self.temporal_norm(x_temporal[:, :, :self.out_features] + out_t)
        x = x_temporal.reshape(batch_size, n_agents, seq_len, -1).permute(0, 2, 1, 3)

        return x


class TGAT(MLP):
    """
    Temporal Graph Attention Network for player velocity completion.
    Processes sequence data with time-aware spatiotemporal attention.

    Input: (seq_len, batch, n_nodes, features)
    Output: velocity prediction for the final timestep
    """

    def __init__(self, params, parser=None):
        super(TGAT, self).__init__(params, parser=None)

        self.rnn_dim = params.get("rnn_dim", self.n_all_agents * 2)
        self.n_layers_rnn = params.get("n_layers", 1)

        self.n_heads = 4
        self.n_tgat_layers = 2
        self.time_dim = 64
        self.d_model = self.n_hid * 2

        self.n_input_add = 0
        self.eventtypes = params["eventtypes"]
        self.ballend = params["ballend"]
        if self.eventtypes:
            self.n_input_add += 1
        if self.ballend:
            self.n_input_add += 1

        # Time encoding
        self.time_encode = TimeEncode(self.time_dim)

        # Input projection
        self.input_proj = nn.Linear(self.n_input + self.n_input_add, self.d_model)

        # Temporal graph attention layers
        self.tgat_layers = nn.ModuleList([
            TemporalGraphAttentionLayer(
                in_features=self.d_model,
                out_features=self.d_model,
                time_dim=self.time_dim,
                n_heads=self.n_heads,
                dropout=self.dropout_prob,
            )
            for _ in range(self.n_tgat_layers)
        ])

        # Feed-forward layers
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.ffn_norm = nn.LayerNorm(self.d_model)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * self.n_all_agents, self.n_hid * self.n_all_agents),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.n_hid * self.n_all_agents, self.n_output * self.n_players),
        )

        if self.eventtypes:
            self.fc_eventtype = nn.Linear(len(EVENTTYPE_CATEGORIES), self.n_all_agents)
        if self.ballend:
            self.fc_ballend = nn.Linear(self.n_input, self.n_all_agents)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def feat_ballend(self, ball_end_loc):
        batch_size = ball_end_loc.size(0)
        ball_end_loc = ball_end_loc.reshape(batch_size, -1).to(self.device).clone()
        feat_ball_end = self.fc_ballend(ball_end_loc).reshape(
            batch_size, self.n_all_agents, -1
        )
        return feat_ball_end

    def feat_eventtype(self, event_type):
        batch_size = event_type.size(0)
        event_type = event_type.reshape(batch_size, -1).to(self.device).clone()
        feat_event_type = self.fc_eventtype(event_type).reshape(
            batch_size, self.n_all_agents, -1
        )
        return feat_event_type

    def compute_dt_sequence(self, frames, seq_len, batch_size, device):
        """
        Compute time differences for each timestep in the sequence.

        Returns: (batch, seq_len) tensor of dt values in seconds
        """
        dt_values = torch.ones(batch_size, seq_len).to(device) * DT

        if frames is not None and frames.size(1) >= 2:
            for t in range(1, seq_len):
                frame_diff = (frames[:, t, 0] - frames[:, t - 1, 0]).float()
                dt_values[:, t] = frame_diff / FRAME_RATE

        return dt_values

    def module(self, x, time_enc):
        """
        x: (batch, seq_len, n_agents, n_input)
        time_enc: (batch, seq_len, time_dim)
        """
        batch_size, seq_len, n_agents, _ = x.size()

        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, n_agents, d_model)

        # Apply temporal graph attention layers
        for layer in self.tgat_layers:
            x = layer(x, time_enc)

        # Feed-forward network
        x = self.ffn_norm(x + self.ffn(x))

        # Take final timestep for output
        x_final = x[:, -1, :, :]  # (batch, n_agents, d_model)

        # Output projection
        x_out = x_final.reshape(batch_size, -1)
        x_out = self.output_proj(x_out).reshape(batch_size, self.n_players, 2)

        return x_out

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")

        seq_len = items["cies"].size(0)
        batch_size = items["cies"].size(1)

        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            # Shape (1, batch_size) for compatibility with MODELS_SEQ concatenation
            batch_losses2["L_rec2"] = torch.zeros(1, batch_size).to(self.device)

        # Build sequence input
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
                feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)

            x_seq.append(x_t_tmp)

        x_seq = torch.stack(x_seq, dim=1)  # (batch, seq_len, n_agents, features)

        # Compute time encoding from dt sequence
        frames = items.get("frames", None)
        dt_values = self.compute_dt_sequence(frames, seq_len, batch_size, self.device)
        time_enc = self.time_encode(dt_values)  # (batch, seq_len, time_dim)

        # Forward pass
        v_t_pred = self.module(x_seq, time_enc)

        # Get ground truth velocity for final timestep
        v_t = items["cies"][
            -1, :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
        ].to(self.device)

        # Loss
        player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
        player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
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
            data_test["v_rb"] = np.zeros((n_sample, batch_size, self.n_all_agents, self.n_input))
            data_test["loc_gt"] = np.zeros((n_sample, batch_size, self.n_all_agents, self.n_input))
            data_test["L_efobso"] = np.zeros((n_sample, batch_size))
            data_test["frames"] = np.zeros((n_sample, batch_size))
            data_test["L_angular"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_x"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_y"] = np.zeros((n_sample, batch_size))
            data_test["L_angular_rb"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_x_rb"] = np.zeros((n_sample, batch_size))
            data_test["L_axis_y_rb"] = np.zeros((n_sample, batch_size))

        # Compute time encoding
        frames = items.get("frames", None)
        dt_values = self.compute_dt_sequence(frames, seq_len, batch_size, self.device)
        time_enc = self.time_encode(dt_values)

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
                    feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
                    x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)

                x_seq.append(x_t_tmp)

            x_seq = torch.stack(x_seq, dim=1)

            # Forward pass
            v_t_pred = self.module(x_seq, time_enc)

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
            player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
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
