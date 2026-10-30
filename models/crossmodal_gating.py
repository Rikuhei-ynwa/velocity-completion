import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .gnn import GNN
from models.utils import add_difference, loss_reconst, angular_error, axis_error

# Cross-modal Gating Network for velocity completion
# Uses gating mechanisms to fuse position and contextual information

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class GatingUnit(nn.Module):
    """
    Gating unit for cross-modal fusion.
    Learns to weight information from different modalities.
    """

    def __init__(self, input_dim, hidden_dim):
        super(GatingUnit, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        """
        Gated fusion of two modalities.

        Parameters
        ----------
        x1 : torch.Tensor
            First modality features
        x2 : torch.Tensor
            Second modality features

        Returns
        -------
        torch.Tensor
            Gated fusion output
        """
        combined = torch.cat([x1, x2], dim=-1)
        gate = self.gate(combined)
        return gate * x1 + (1 - gate) * x2


class CrossModalEncoder(nn.Module):
    """
    Encoder with cross-modal gating between position and context features.
    """

    def __init__(self, pos_dim, ctx_dim, hidden_dim, n_agents):
        super(CrossModalEncoder, self).__init__()
        self.n_agents = n_agents

        # Position encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Context encoder (event type, ball end position, etc.)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-modal gating
        self.pos_to_ctx_gate = GatingUnit(hidden_dim, hidden_dim)
        self.ctx_to_pos_gate = GatingUnit(hidden_dim, hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, pos_features, ctx_features):
        """
        Parameters
        ----------
        pos_features : torch.Tensor
            Position features (batch, n_agents, pos_dim)
        ctx_features : torch.Tensor
            Context features (batch, n_agents, ctx_dim)

        Returns
        -------
        torch.Tensor
            Fused features (batch, n_agents, hidden_dim)
        """
        # Encode each modality
        pos_enc = self.pos_encoder(pos_features)
        ctx_enc = self.ctx_encoder(ctx_features)

        # Cross-modal gating
        pos_gated = self.pos_to_ctx_gate(pos_enc, ctx_enc)
        ctx_gated = self.ctx_to_pos_gate(ctx_enc, pos_enc)

        # Fusion
        fused = torch.cat([pos_gated, ctx_gated], dim=-1)
        output = self.fusion(fused)

        return output


class CrossModalGating(GNN):
    """
    Cross-modal Gating Network for player velocity completion.
    Uses gating mechanisms to fuse position, event type, and ball end features.
    """

    def __init__(self, params, parser=None):
        super(CrossModalGating, self).__init__(params, parser=None)

        self.n_input_add = 0
        self.eventtypes = params["eventtypes"]
        self.ballend = params["ballend"]
        if self.eventtypes:
            self.n_input_add += 1
        if self.ballend:
            self.n_input_add += 1

        # Context dimension (event type + ball end)
        self.ctx_dim = self.n_input_add if self.n_input_add > 0 else 1

        # Cross-modal encoder
        self.cross_modal_encoder = CrossModalEncoder(
            pos_dim=self.n_input,
            ctx_dim=self.ctx_dim,
            hidden_dim=self.n_hid,
            n_agents=self.n_all_agents,
        )

        # Graph convolution layers
        self.fc1 = nn.Linear(self.n_hid, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid * 2 + self.n_edge_type, self.n_hid)
        self.fc3 = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_output * self.n_players,
        )
        self.bn = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        if self.eventtypes:
            self.fc_eventtype = nn.Linear(len(EVENTTYPE_CATEGORIES), self.n_all_agents)
        if self.ballend:
            self.fc_ballend = nn.Linear(self.n_input, self.n_all_agents)

        self.init_weights()

    def get_context_features(self, items, batch_size):
        """
        Extract and combine context features.
        """
        ctx_list = []
        if self.ballend:
            feat_ball_end = self.feat_ballend(items["cies"][:, 1:2, :])
            ctx_list.append(feat_ball_end)
        if self.eventtypes:
            feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
            ctx_list.append(feat_event_type)

        if len(ctx_list) > 0:
            ctx_features = torch.cat(ctx_list, dim=2)
        else:
            # Dummy context if none provided
            ctx_features = torch.zeros(batch_size, self.n_all_agents, 1).to(self.device)

        return ctx_features

    def module(self, pos_features, ctx_features, edge_type, node_type):
        batch_size = pos_features.size(0)

        # Cross-modal encoding
        x = self.cross_modal_encoder(pos_features, ctx_features)

        # Graph convolution
        x = self.fc1(x)
        x = F.elu(x)
        x = self.batch_norm(x)

        x = self.node2edge(x, self.rel_receiver, self.rel_sender, edge_type)
        x = self.fc2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        x = self.edge2node(x, self.rel_receiver, node_type)
        x = x.reshape(batch_size, -1)
        x = self.fc3(x).reshape(batch_size, self.n_players, 2)

        return x

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batch_size = items["cies"].size(0)

        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(batch_size).to(self.device)

        # Position features
        x_t = torch.cat(
            [items["cies"][:, 0:1, :], items["cies"][:, 2 : len(LOC_NODES), :]],
            dim=1,
        ).to(self.device)
        v_t = items["cies"][:, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :].to(
            self.device
        )
        edge_type = items["edge_type"][:, :, :].to(self.device)
        node_type = items["node_type"][:, :, :].to(self.device)

        # Context features
        ctx_features = self.get_context_features(items, batch_size)

        # Forward pass
        v_t_pred = self.module(x_t, ctx_features, edge_type, node_type)

        # Loss
        player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
        player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
        batch_losses["L_rec"] += loss_reconst(
            player_vel_pred, player_vel_GT, self.n_players, Sum=True
        )
        if acc == 0:
            batch_losses2["L_rec2"] += loss_reconst(
                player_vel_pred, player_vel_GT, self.n_players, Sum=False
            )

        return batch_losses, batch_losses2

    def sample(self, items, players_info, n_sample=1, TEST=False):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batch_size = items["cies"].size(0)

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

        for i in range(n_sample):
            x_t = torch.cat(
                [cies_n[i][:, 0:1, :], cies_n[i][:, 2 : len(LOC_NODES), :]],
                dim=1,
            ).to(self.device)
            v_t = cies_n[i][
                :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
            ].to(self.device)
            edge_type = items["edge_type"][:, :, :].to(self.device)
            node_type = items["node_type"][:, :, :].to(self.device)

            # Context features
            ctx_features = self.get_context_features(items, batch_size)

            # Forward pass
            v_t_pred = self.module(x_t, ctx_features, edge_type, node_type)

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

            pos_GT = x_t.detach().cpu().numpy()
            max1 = cfg.FIELD_LENGTH / 2
            max2 = cfg.FIELD_WIDTH / 2
            pos_GT[:, :, 0] = pos_GT[:, :, 0] * max1
            pos_GT[:, :, 1] = pos_GT[:, :, 1] * max2

            if TEST:
                ball_vel_rb = np.expand_dims(v_t[:, 0, :].detach().cpu().numpy(), axis=1)
                player_vel_rb = cies_n[i][:, len(LOC_NODES + VEL_NODES) :, :].to(
                    self.device
                )
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
                data_test["frames"][i] += items["frames"][:, :].detach().cpu().numpy().T[0]
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
