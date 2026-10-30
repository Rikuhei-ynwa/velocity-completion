import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .gat import GAT, GraphAttentionLayer
from .rnn import RNN
from models.utils import add_difference, loss_reconst, angular_error, axis_error

# GAT + RNN for velocity completion
# Combines Graph Attention Networks with Recurrent Neural Networks

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class GATRNN(GAT, RNN):
    """
    GAT + RNN model for player velocity completion.
    Uses Graph Attention for spatial modeling and GRU for temporal modeling.
    """

    def __init__(self, params, parser=None):
        super(GATRNN, self).__init__(params, parser=parser)

        # RNN
        self.rnn = nn.GRU(
            (self.n_hid + self.n_input) * self.n_all_agents,
            self.rnn_dim,
            self.n_layers,
        )
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)

        # Prior (GAT-based)
        self.n_input_pri = (
            self.n_input + int(self.rnn_dim / self.n_all_agents)
        ) * self.n_layers
        self.gat1_pri = GraphAttentionLayer(
            self.n_input_pri,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=True,
        )
        self.gat2_pri = GraphAttentionLayer(
            self.n_hid * self.n_heads,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=False,
        )
        self.fc_pri = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_hid * self.n_all_agents,
        )
        # BatchNorm after gat1_pri: output size is n_hid * n_heads (due to concat=True)
        self.bn_pri = nn.BatchNorm1d(self.n_hid * self.n_heads * self.n_all_agents)

        # Decoder (GAT-based)
        self.n_input_dec = (
            (self.n_hid + self.n_input_add + int(self.rnn_dim / self.n_all_agents))
            * self.n_layers
        )
        self.gat1_dec = GraphAttentionLayer(
            self.n_input_dec,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=True,
        )
        self.gat2_dec = GraphAttentionLayer(
            self.n_hid * self.n_heads,
            self.n_hid,
            n_heads=self.n_heads,
            dropout=self.dropout_prob,
            concat=False,
        )
        self.fc_dec = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_output * self.n_players,
        )
        # BatchNorm after gat1_dec: output size is n_hid * n_heads (due to concat=True)
        self.bn_dec = nn.BatchNorm1d(self.n_hid * self.n_heads * self.n_all_agents)

        if self.eventtypes:
            self.fc_eventtype = nn.Linear(len(EVENTTYPE_CATEGORIES), self.n_all_agents)
        if self.ballend:
            self.fc_ballend = nn.Linear(self.n_input, self.n_all_agents)

        self.init_weights()

    def batch_norm_module(self, x, module):
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        if module == "pri":
            x = self.bn_pri(x)
        elif module == "dec":
            x = self.bn_dec(x)
        x = x.reshape(batch_size, self.n_all_agents, -1)
        return x

    def Prior(self, x, h, node_type):
        """
        Prior network using GAT.
        """
        batch_size = x.size(1)
        input_combined = torch.cat([x, h], dim=3).reshape(batch_size, self.n_all_agents, -1)

        x = self.gat1_pri(input_combined)
        x = F.elu(x)
        x = self.batch_norm_module(x, "pri")

        x = self.gat2_pri(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        # Concatenate node type
        x = torch.cat([x, node_type], dim=2)
        x = x.reshape(batch_size, -1)
        x = self.fc_pri(x).reshape(batch_size, self.n_all_agents, -1)

        return x

    def Decoder(self, z, h, node_type):
        """
        Decoder network using GAT.
        """
        batch_size = z.size(1)
        x = torch.cat([z, h], dim=3).reshape(batch_size, self.n_all_agents, -1)

        x = self.gat1_dec(x)
        x = F.elu(x)
        x = self.batch_norm_module(x, "dec")

        x = self.gat2_dec(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)

        # Concatenate node type
        x = torch.cat([x, node_type], dim=2)
        x = x.reshape(batch_size, -1)
        x = self.fc_dec(x).reshape(batch_size, self.n_players, -1)

        return x

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batch_size = items["cies"].size(1)
        seq_len = items["cies"].size(0)

        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(seq_len, batch_size).to(self.device)

        # Initialize hidden state
        h_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(self.device)

        for t in range(seq_len):
            x_t = torch.cat(
                [
                    items["cies"][t, :, 0:1, :],
                    items["cies"][t, :, 2 : len(LOC_NODES), :],
                ],
                dim=1,
            ).to(self.device)
            v_t = items["cies"][
                t, :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
            ].to(self.device)
            node_type = items["node_type"][t, :, :, :].to(self.device)

            h_t_1 = h_t.reshape(h_t.size(0), batch_size, self.n_all_agents, -1)
            x_t_tmp = x_t.unsqueeze(0).expand(h_t_1.size(0), -1, -1, -1)

            # Prior
            z_t = self.Prior(x_t_tmp, h_t_1, node_type)
            z_t_tmp = z_t.unsqueeze(0).expand(h_t_1.size(0), -1, -1, -1)

            if t < seq_len - 1:
                # RNN
                _, h_t = self.rnn(
                    torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(
                        h_t.size(0), batch_size, -1
                    ),
                    h_t,
                )
            elif t == seq_len - 1:
                # Add features for decoder
                if self.ballend:
                    feat_ball_end = (
                        self.feat_ballend(items["cies"][t, :, 1:2, :])
                        .unsqueeze(0)
                        .expand(h_t.size(0), -1, -1, -1)
                    )
                    z_t_tmp = torch.cat([z_t_tmp, feat_ball_end], dim=3)
                if self.eventtypes:
                    feat_event_type = (
                        self.feat_eventtype(items["eventtypes"][t, :, :, :])
                        .unsqueeze(0)
                        .expand(h_t.size(0), -1, -1, -1)
                    )
                    z_t_tmp = torch.cat([z_t_tmp, feat_event_type], dim=3)

                # Decoder
                v_t_pred = self.Decoder(z_t_tmp, h_t_1, node_type)

                # Loss
                player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
                player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
                batch_losses["L_rec"] += loss_reconst(
                    player_vel_pred, player_vel_GT, self.n_players, Sum=True
                )
                if acc == 0:
                    batch_losses2["L_rec2"][t] += loss_reconst(
                        player_vel_pred, player_vel_GT, self.n_players, Sum=False
                    )

        return batch_losses, batch_losses2

    def sample(self, items, players_info, n_sample=1, TEST=False):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batch_size = items["cies"].size(1)
        seq_len = items["cies"].size(0)

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
            h_t = torch.zeros(self.n_layers, batch_size, self.rnn_dim).to(self.device)

            for t in range(seq_len):
                x_t = torch.cat(
                    [
                        cies_n[i][t, :, 0:1, :],
                        cies_n[i][t, :, 2 : len(LOC_NODES), :],
                    ],
                    dim=1,
                ).to(self.device)
                v_t = cies_n[i][
                    t, :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
                ].to(self.device)
                node_type = items["node_type"][t, :, :, :].to(self.device)

                h_t_1 = h_t.reshape(h_t.size(0), batch_size, self.n_all_agents, -1)
                x_t_tmp = x_t.unsqueeze(0).expand(h_t_1.size(0), -1, -1, -1)

                # Prior
                z_t = self.Prior(x_t_tmp, h_t_1, node_type)
                z_t_tmp = z_t.unsqueeze(0).expand(h_t_1.size(0), -1, -1, -1)

                if t < seq_len - 1:
                    _, h_t = self.rnn(
                        torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(
                            h_t.size(0), batch_size, -1
                        ),
                        h_t,
                    )
                elif t == seq_len - 1:
                    if self.ballend:
                        feat_ball_end = (
                            self.feat_ballend(items["cies"][t, :, 1:2, :])
                            .unsqueeze(0)
                            .expand(h_t.size(0), -1, -1, -1)
                        )
                        z_t_tmp = torch.cat([z_t_tmp, feat_ball_end], dim=3)
                    if self.eventtypes:
                        feat_event_type = (
                            self.feat_eventtype(items["eventtypes"][t, :, :, :])
                            .unsqueeze(0)
                            .expand(h_t.size(0), -1, -1, -1)
                        )
                        z_t_tmp = torch.cat([z_t_tmp, feat_event_type], dim=3)

                    v_t_pred = self.Decoder(z_t_tmp, h_t_1, node_type)

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
                        ball_vel_rb = np.expand_dims(
                            v_t[:, 0, :].detach().cpu().numpy(), axis=1
                        )
                        player_vel_rb = cies_n[i][
                            t, :, len(LOC_NODES + VEL_NODES) :, :
                        ].to(self.device)
                        data_test["v_pred"][i] = player_vel_pred.detach().cpu().numpy()
                        data_test["v_gt"][i] = player_vel_GT.detach().cpu().numpy()
                        data_test["v_rb"][i] = np.concatenate(
                            (ball_vel_rb, player_vel_rb.detach().cpu().numpy()), axis=1
                        )
                        data_test["loc_gt"][i] = pos_GT
                        data_test["L_efobso"][i] += (
                            loss_reconst(
                                player_vel_rb, player_vel_GT, self.n_players, Sum=False
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["L_angular"][i] += (
                            angular_error(
                                player_vel_pred, player_vel_GT, self.n_players, Sum=False
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["L_axis_x"][i] += (
                            axis_error(
                                player_vel_pred,
                                player_vel_GT,
                                self.n_players,
                                axis="x",
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["L_axis_y"][i] += (
                            axis_error(
                                player_vel_pred,
                                player_vel_GT,
                                self.n_players,
                                axis="y",
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["frames"][i] += (
                            items["frames"][:, t, :].detach().cpu().numpy().T[0]
                        )
                        data_test["L_angular_rb"][i] += (
                            angular_error(
                                player_vel_rb, player_vel_GT, self.n_players, Sum=False
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["L_axis_x_rb"][i] += (
                            axis_error(
                                player_vel_rb,
                                player_vel_GT,
                                self.n_players,
                                axis="x",
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        data_test["L_axis_y_rb"][i] += (
                            axis_error(
                                player_vel_rb,
                                player_vel_GT,
                                self.n_players,
                                axis="y",
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
