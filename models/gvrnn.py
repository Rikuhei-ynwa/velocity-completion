import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .gnn import GNN
from .vae import VAE
from .rnn import RNN
from models.utils import (
    add_difference,
    sample_gauss,
    kld_gauss,
    loss_reconst,
    angular_error,
    axis_error,
)

# Rikuhei Umemoto, 2024
# modifying the code
# https://github.com/ezhan94/multiagent-programmatic-supervision
# https://github.com/keisuke198619/C-OBSO

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class GVRNN(GNN, VAE, RNN):
    def __init__(self, params, parser=None):
        super(GVRNN, self).__init__(params, parser=parser)

        self.fixedsigma = False

        # RNN
        self.rnn = nn.GRU(
            (self.n_hid + self.n_input) * self.n_all_agents,
            self.rnn_dim,
            self.n_layers,
        )
        nn.init.xavier_normal_(self.rnn.weight_hh_l0)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0)

        # Prior
        self.n_input_pri = (
            (self.n_input + int(self.rnn_dim / self.n_all_agents))
            * self.n_layers
            )
        self.fc1_pri = nn.Linear(self.n_input_pri, self.n_hid)
        self.fc2_pri = nn.Linear(self.n_hid * 2 + self.n_edge_type, self.n_hid)
        self.fc3_pri = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_hid * self.n_all_agents,
        )
        self.bn_pri = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        # Encoder
        self.n_input_enc = (
            (self.n_input * 2 + int(self.rnn_dim / self.n_all_agents))
            * self.n_layers
            )
        self.fc1_enc = nn.Linear(self.n_input_enc, self.n_hid)
        self.fc2_enc = nn.Linear(self.n_hid * 2 + self.n_edge_type, self.n_hid)
        self.fc3_enc = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_hid * self.n_all_agents,
        )
        self.bn_enc = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        # Decoder
        self.n_input_dec = (
            (self.n_hid + self.n_input_add + int(self.rnn_dim / self.n_all_agents))
            * self.n_layers
            )
        self.fc1_dec = nn.Linear(self.n_input_dec, self.n_hid)
        self.fc2_dec = nn.Linear(self.n_hid * 2 + self.n_edge_type, self.n_hid)
        self.fc3_dec = nn.Linear(
            (self.n_hid + self.n_node_type) * self.n_all_agents,
            self.n_output * self.n_players,
        )
        self.bn_dec = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        ## Definition of the distributions
        # Prior
        self.prior_mean = nn.Linear(self.n_hid, self.n_hid)
        self.prior_std = nn.Sequential(
            nn.Linear(self.n_hid, self.n_hid),
            nn.Softplus(),
        )
        self.prior_std.apply(self.weights_init)

        # Encoder
        self.enc_mean = nn.Linear(self.n_hid, self.n_hid)
        self.enc_std = nn.Sequential(
            nn.Linear(self.n_hid, self.n_hid),
            nn.Softplus(),
        )
        self.enc_std.apply(self.weights_init)

        # Decoder
        self.dec_mean = nn.Linear(self.n_output, self.n_output)
        if not self.fixedsigma:
            self.dec_std = nn.Sequential(
                nn.Linear(self.n_output, self.n_output),
                nn.Softplus(),
            )
            self.dec_std.apply(self.weights_init)

        self.init_weights()

    def batch_norm(self, x, module):
        x = x.reshape(x.size(0), -1)
        if module == "pri":
            x = self.bn_pri(x)
        elif module == "enc":
            x = self.bn_enc(x)
        elif module == "dec":
            x = self.bn_dec(x)
        x = x.reshape(x.size(0), self.n_all_agents, -1)
        return x

    def Prior(self, x, h, edge_type, node_type):
        _batchsize = x.size(1)
        x = torch.cat([x, h], dim=3).reshape(_batchsize, self.n_all_agents, -1)
        x = self.fc1_pri(x)
        x = F.elu(x)
        x = self.batch_norm(x, "pri")
        x = self.node2edge(x, self.rel_receiver, self.rel_sender, edge_type)
        x = self.fc2_pri(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.edge2node(x, self.rel_receiver, node_type)
        x = x.reshape(_batchsize, -1)
        x = self.fc3_pri(x).reshape(_batchsize, self.n_all_agents, -1)

        mean = self.prior_mean(x)
        std = self.prior_std(x)

        del x, h, edge_type, node_type

        return mean, std

    def Encoder(self, x, v, h, edge_type, node_type):
        _batchsize = x.size(1)
        x = torch.cat([x, v, h], dim=3).reshape(_batchsize, self.n_all_agents, -1)
        x = self.fc1_enc(x)
        x = F.elu(x)
        x = self.batch_norm(x, "enc")
        x = self.node2edge(x, self.rel_receiver, self.rel_sender, edge_type)
        x = self.fc2_enc(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.edge2node(x, self.rel_receiver, node_type)
        x = x.reshape(_batchsize, -1)
        x = self.fc3_enc(x).reshape(_batchsize, self.n_all_agents, -1)

        mean = self.enc_mean(x)
        std = self.enc_std(x)

        del x, v, h, edge_type, node_type

        return mean, std

    def Decoder(self, z, h, edge_type, node_type):
        _batchsize = z.size(1)
        x = torch.cat([z, h], dim=3).reshape(_batchsize, self.n_all_agents, -1)
        x = self.fc1_dec(x)
        x = F.elu(x)
        x = self.batch_norm(x, "dec")
        x = self.node2edge(x, self.rel_receiver, self.rel_sender, edge_type)
        x = self.fc2_dec(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.edge2node(x, self.rel_receiver, node_type)
        x = x.reshape(_batchsize, -1)
        x = self.fc3_dec(x).reshape(_batchsize, self.n_players, -1)

        mean = self.dec_mean(x)
        if not self.fixedsigma:
            std = self.dec_std(x)

        del x, z, h, edge_type, node_type

        return mean, std

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batchSize = items["cies"].size(1)
        seq_len = items["cies"].size(0)

        # The dictionary for storing loss per batch is batch_losses
        # The dictionary for storing loss per event is batch_losses2
        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(seq_len, batchSize).to(self.device)

        # Initialize the hidden state
        h_t = torch.zeros(
            self.n_layers,
            batchSize,
            self.rnn_dim,
        ).to(self.device)

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
            edge_type = items["edge_type"][t, :, :, :].to(self.device)
            node_type = items["node_type"][t, :, :, :].to(self.device)

            # Training
            h_t_1 = h_t.reshape(h_t.size(0), batchSize, self.n_all_agents, -1)
            x_t_tmp = (
                x_t
                .unsqueeze(0)
                .expand(h_t_1.size(0), -1, -1, -1)
                )
            v_t_tmp = (
                v_t
                .unsqueeze(0)
                .expand(h_t_1.size(0), -1, -1, -1)
                )
            # Prior
            # Input: x^t, h^(t-1)
            mean_pri, std_pri = self.Prior(x_t_tmp, h_t_1, edge_type, node_type)

            # Encoder
            # Input: x^t, v^t, h^(t-1)
            mean_enc, std_enc = self.Encoder(x_t_tmp, v_t_tmp, h_t_1, edge_type, node_type)

            # sample z^t
            z_t = (
                sample_gauss(mean_enc, std_enc)
                .reshape(batchSize, self.n_all_agents, -1)
                )
            z_t_tmp = (
                z_t
                .unsqueeze(0)
                .expand(h_t.size(0), -1, -1, -1)
                )

            # Decoder
            # Input: z^t, h^(t-1), MLP(ball_end_loc), MLP(event_type)
            z_t_tmp_tmp = z_t_tmp.clone()
            if self.ballend:
                feat_ball_end = (
                    self.feat_ballend(items["cies"][t, :, 1:2, :])
                    .unsqueeze(0)
                    .expand(h_t.size(0), -1, -1, -1)
                    )
                z_t_tmp_tmp = torch.cat([z_t_tmp_tmp, feat_ball_end], dim=3)
                del feat_ball_end
            if self.eventtypes:
                feat_event_type = (
                    self.feat_eventtype(items["eventtypes"][t, :, :, :])
                    .unsqueeze(0)
                    .expand(h_t.size(0), -1, -1, -1)
                    )
                z_t_tmp_tmp = torch.cat([z_t_tmp_tmp, feat_event_type], dim=3)
                del feat_event_type
            mean_dec, _ = self.Decoder(z_t_tmp_tmp, h_t_1, edge_type, node_type)

            # Loss
            # KLD
            batch_losses["L_kl"] += kld_gauss(mean_enc, std_enc, mean_pri, std_pri)
            # Reconstruction
            player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
            player_vel_pred = add_difference(self, self.restore(mean_dec, players_info), items, t, seq_len, player_vel_GT)
            batch_losses["L_rec"] += loss_reconst(
                player_vel_pred,
                player_vel_GT,
                self.n_players,
                Sum=True,
            )
            if acc == 0:
                batch_losses2["L_rec2"][t] += loss_reconst(
                    player_vel_pred,
                    player_vel_GT,
                    self.n_players,
                    Sum=False,
                )

            # RNN
            # h^t = f(x^t, z^t, h^(t-1))
            _, h_t = self.rnn(
                torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(h_t.size(0), batchSize, -1),
                h_t
                )

            del (
                x_t,
                v_t,
                edge_type,
                node_type,
                x_t_tmp,
                v_t_tmp,
                h_t_1,
                mean_pri,
                std_pri,
                mean_enc,
                std_enc,
                z_t,
                z_t_tmp,
                z_t_tmp_tmp,
                mean_dec,
                player_vel_GT,
                player_vel_pred,
            )

        for key in batch_losses.keys():
            batch_losses[key] /= seq_len

        return batch_losses, batch_losses2

    def sample(self, items, players_info, n_sample=1, TEST=False):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batchSize = items["cies"].size(1)
        seq_len = items["cies"].size(0)
        if n_sample > 1:
            cies_n = [items["cies"].clone() for _ in range(n_sample)]
        else:
            cies_n = [items["cies"].clone().clone()]

        # The dictionary for storing loss per batch is batch_losses
        # The dictionary for storing loss per event is batch_losses2
        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_rec"] = torch.zeros(n_sample).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(n_sample, batchSize).to(self.device)

        if TEST:
            # added
            data_test = {}
            data_test["v_pred"] = np.zeros(
                (n_sample, batchSize, self.n_players, 2),
            )
            data_test["v_gt"] = np.zeros(
                (n_sample, batchSize, self.n_players, 2),
            )
            data_test["v_rb"] = np.zeros(
                (n_sample, batchSize, self.n_all_agents, self.n_input),
            )
            data_test["loc_gt"] = np.zeros(
                (n_sample, batchSize, self.n_all_agents, self.n_input),
            )
            data_test["L_efobso"] = np.zeros(
                (n_sample, batchSize),
            )
            data_test["frames"] = np.zeros(
                (n_sample, batchSize),
            )
            # Angular error (degrees) and axis-specific errors
            data_test["L_angular"] = np.zeros(
                (n_sample, batchSize),
            )
            data_test["L_axis_x"] = np.zeros(
                (n_sample, batchSize),
            )
            data_test["L_axis_y"] = np.zeros(
                (n_sample, batchSize),
            )
            # Angular error and axis-specific errors for rule-based velocity (v_rb)
            data_test["L_angular_rb"] = np.zeros(
                (n_sample, batchSize),
            )
            data_test["L_axis_x_rb"] = np.zeros(
                (n_sample, batchSize),
            )
            data_test["L_axis_y_rb"] = np.zeros(
                (n_sample, batchSize),
            )

        for i in range(n_sample):
            # Initialize the hidden state
            h_t = torch.zeros(
                self.n_layers,
                batchSize,
                self.rnn_dim,
            ).to(self.device)

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
                edge_type = items["edge_type"][t, :, :, :].to(self.device)
                node_type = items["node_type"][t, :, :, :].to(self.device)

                # Validation and Test
                h_t_1 = h_t.reshape(h_t.size(0), batchSize, self.n_all_agents, -1)
                x_t_tmp = (
                    x_t
                    .unsqueeze(0)
                    .expand(h_t_1.size(0), -1, -1, -1)
                    )
                # Prior
                # Input: x^t, h^(t-1)
                mean_pri, std_pri = self.Prior(x_t_tmp, h_t_1, edge_type, node_type)

                # sample z^t
                z_t = (
                    sample_gauss(mean_pri, std_pri)
                    .reshape(batchSize, self.n_all_agents, -1)
                    )
                z_t_tmp = (
                    z_t
                    .unsqueeze(0)
                    .expand(h_t_1.size(0), -1, -1, -1)
                    )

                if t < seq_len - 1:
                    # RNN
                    # h^t = f(x^t, v^t, z^t, h^(t-1))
                    _, h_t = self.rnn(
                        (torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(h_t.size(0), batchSize, -1)),
                        h_t
                        )
                    del (
                        x_t,
                        v_t,
                        edge_type,
                        node_type,
                        x_t_tmp,
                        h_t_1,
                        mean_pri,
                        std_pri,
                        z_t,
                    )

                elif t == seq_len - 1:
                    # Decoder
                    # Input: z^t, h^(t-1), MLP(ball_end_loc), MLP(event_type)
                    if self.ballend:
                        feat_ball_end = (
                            self.feat_ballend(items["cies"][t, :, 1:2, :])
                            .unsqueeze(0)
                            .expand(h_t.size(0), -1, -1, -1)
                            )
                        z_t_tmp = torch.cat([z_t_tmp, feat_ball_end], dim=3)
                        del feat_ball_end
                    if self.eventtypes:
                        feat_event_type = (
                            self.feat_eventtype(items["eventtypes"][t, :, :, :])
                            .unsqueeze(0)
                            .expand(h_t.size(0), -1, -1, -1)
                            )
                        z_t_tmp = torch.cat([z_t_tmp, feat_event_type], dim=3)
                        del feat_event_type
                    mean_dec, _ = self.Decoder(z_t_tmp, h_t_1, edge_type, node_type)

                    # Loss
                    # Reconstruction
                    player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
                    player_vel_pred = add_difference(self, self.restore(mean_dec, players_info), items)
                    batch_losses["L_rec"][i] += loss_reconst(
                        player_vel_pred,
                        player_vel_GT,
                        self.n_players,
                        Sum=True,
                    )
                    if acc == 0:
                        batch_losses2["L_rec2"][i] += loss_reconst(
                            player_vel_pred,
                            player_vel_GT,
                            self.n_players,
                            Sum=False,
                        )

                    # position
                    pos_GT = x_t.detach().cpu().numpy()
                    max1 = cfg.FIELD_LENGTH / 2
                    max2 = cfg.FIELD_WIDTH / 2
                    pos_GT[:, :, 0] = pos_GT[:, :, 0] * max1
                    pos_GT[:, :, 1] = pos_GT[:, :, 1] * max2

                    if TEST:
                        ball_vel_rb = np.expand_dims(
                            v_t[:, 0, :].detach().cpu().numpy(),
                            axis=1
                            )
                        player_vel_rb = cies_n[i][
                            t, :, len(LOC_NODES + VEL_NODES) :, :
                            ].to(self.device)
                        data_test["v_pred"][i] = player_vel_pred.detach().cpu().numpy()
                        data_test["v_gt"][i] = player_vel_GT.detach().cpu().numpy()
                        data_test["v_rb"][i] = np.concatenate(
                            (ball_vel_rb, player_vel_rb.detach().cpu().numpy()),
                            axis=1
                            )
                        data_test["loc_gt"][i] = pos_GT
                        data_test["L_efobso"][i] += (
                            loss_reconst(
                                player_vel_rb,
                                player_vel_GT,
                                self.n_players,
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # Angular error (degrees)
                        data_test["L_angular"][i] += (
                            angular_error(
                                player_vel_pred,
                                player_vel_GT,
                                self.n_players,
                                Sum=False,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        # Axis-specific errors (squared error for RMSE)
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
                            items["frames"][:,t,:].detach().cpu().numpy().T[0]
                        )
                        # Angular error and axis-specific errors for rule-based velocity (v_rb)
                        data_test["L_angular_rb"][i] += (
                            angular_error(
                                player_vel_rb,
                                player_vel_GT,
                                self.n_players,
                                Sum=False,
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
                        del ball_vel_rb, player_vel_rb

                    del (
                        x_t,
                        v_t,
                        edge_type,
                        node_type,
                        x_t_tmp,
                        h_t_1,
                        z_t,
                        z_t_tmp,
                        mean_dec,
                        player_vel_GT,
                        player_vel_pred,
                        pos_GT,
                    )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
