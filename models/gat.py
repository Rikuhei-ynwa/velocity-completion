import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .mlp import MLP
from models.utils import add_difference, loss_reconst, angular_error, axis_error

# Graph Attention Network for velocity completion
# Based on "Graph Attention Networks" (Veličković et al., 2018)

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with multi-head attention.
    """

    def __init__(self, in_features, out_features, n_heads=4, dropout=0.0, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat

        # Linear transformation for each head
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.zeros(n_heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, out_features))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_type=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch, n_agents, in_features)
        edge_type : torch.Tensor, optional
            Edge type tensor for additional edge features

        Returns
        -------
        torch.Tensor
            Output features (batch, n_agents, out_features * n_heads or out_features)
        """
        batch_size, n_agents, _ = x.size()

        # Linear transformation: (batch, n_agents, out_features * n_heads)
        h = self.W(x)
        h = h.view(batch_size, n_agents, self.n_heads, self.out_features)

        # Compute attention scores
        # (batch, n_agents, n_heads, out_features)
        attn_src = (h * self.a_src).sum(dim=-1)  # (batch, n_agents, n_heads)
        attn_dst = (h * self.a_dst).sum(dim=-1)  # (batch, n_agents, n_heads)

        # Broadcast for pairwise attention
        # (batch, n_agents, 1, n_heads) + (batch, 1, n_agents, n_heads)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leaky_relu(attn)  # (batch, n_agents, n_agents, n_heads)

        # Apply softmax
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        # Apply attention to values
        # h: (batch, n_agents, n_heads, out_features)
        # attn: (batch, n_agents, n_agents, n_heads)
        h = h.permute(0, 2, 1, 3)  # (batch, n_heads, n_agents, out_features)
        attn = attn.permute(0, 3, 1, 2)  # (batch, n_heads, n_agents, n_agents)
        out = torch.matmul(attn, h)  # (batch, n_heads, n_agents, out_features)

        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous()
            out = out.view(batch_size, n_agents, -1)
        else:
            out = out.mean(dim=1)  # Average over heads

        return out


class GAT(MLP):
    """
    Graph Attention Network for player velocity completion.
    """

    def __init__(self, params, parser=None):
        super(GAT, self).__init__(params, parser=None)

        self.n_edge_type = params["n_edge_type"]
        self.n_node_type = params["n_node_type"]
        self.n_heads = 4

        self.n_input_add = 0
        self.eventtypes = params["eventtypes"]
        self.ballend = params["ballend"]
        if self.eventtypes:
            self.n_input_add += 1
        if self.ballend:
            self.n_input_add += 1

        # GAT layers
        self.gat1 = GraphAttentionLayer(
            self.n_input + self.n_input_add,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=True,
        )
        self.gat2 = GraphAttentionLayer(
            self.n_hid * self.n_heads,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=False,
        )

        # Output layer
        self.fc_out = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_output * self.n_players,
        )
        # BatchNorm after gat1: output size is n_hid * n_heads (due to concat=True)
        self.bn = nn.BatchNorm1d(self.n_hid * self.n_heads * self.n_all_agents)

        if self.eventtypes:
            self.fc_eventtype = nn.Linear(len(EVENTTYPE_CATEGORIES), self.n_all_agents)
        if self.ballend:
            self.fc_ballend = nn.Linear(self.n_input, self.n_all_agents)

        self.init_weights()

    def batch_norm(self, x):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.bn(x)
        x = x.reshape(batch_size, self.n_all_agents, -1)
        return x

    def module(self, x, edge_type, node_type):
        batch_size = x.size(0)

        # GAT layers
        x = self.gat1(x)
        x = F.elu(x)
        x = self.batch_norm(x)

        x = self.gat2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        # Concatenate node type
        x = torch.cat([x, node_type], dim=2)

        # Output
        x = x.reshape(batch_size, -1)
        x = self.fc_out(x).reshape(batch_size, self.n_players, 2)

        return x

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

        x_t = torch.cat(
            [items["cies"][:, 0:1, :], items["cies"][:, 2 : len(LOC_NODES), :]],
            dim=1,
        ).to(self.device)
        x_t_tmp = x_t.clone()
        v_t = items["cies"][:, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :].to(
            self.device
        )
        edge_type = items["edge_type"][:, :, :].to(self.device)
        node_type = items["node_type"][:, :, :].to(self.device)

        if self.ballend:
            feat_ball_end = self.feat_ballend(items["cies"][:, 1:2, :])
            x_t_tmp = torch.cat([x_t_tmp, feat_ball_end], dim=2)
        if self.eventtypes:
            feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
            x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)

        # Forward pass
        v_t_pred = self.module(x_t_tmp, edge_type, node_type)

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
            x_t_tmp = x_t.clone()
            v_t = cies_n[i][
                :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
            ].to(self.device)
            edge_type = items["edge_type"][:, :, :].to(self.device)
            node_type = items["node_type"][:, :, :].to(self.device)

            if self.ballend:
                feat_ball_end = self.feat_ballend(cies_n[i][:, 1:2, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_ball_end], dim=2)
            if self.eventtypes:
                feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)

            # Forward pass
            v_t_pred = self.module(x_t_tmp, edge_type, node_type)

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
