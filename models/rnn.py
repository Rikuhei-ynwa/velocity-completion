import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .mlp import MLP
from models.utils import add_difference, loss_reconst, angular_error, axis_error

# Rikuhei Umemoto, 2024
# modifying the code
# https://github.com/ezhan94/multiagent-programmatic-supervision
# https://github.com/keisuke198619/C-OBSO

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE

# Frame rate of tracking data (Hz)
FRAME_RATE = 25.0
DT = 1.0 / FRAME_RATE  # Default time between frames in seconds


class RNN(MLP):
    def __init__(self, params, parser=None):
        super(RNN, self).__init__(params, parser=parser)

        self.rnn_dim = params["rnn_dim"]
        self.n_layers = params["n_layers"]
        self.fixedsigma = False

        # Time difference (dt) as input feature
        self.use_dt = params.get("use_dt", False)
        self.dt_dim = 1 if self.use_dt else 0  # Dimension for dt feature

        # RNN input size: add dt_dim if use_dt is True
        rnn_input_size = (self.n_hid + self.n_input) * self.n_all_agents + self.dt_dim

        # RNN
        self.rnn = nn.GRU(
            rnn_input_size,
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
        self.fc2_pri = nn.Linear(self.n_hid, self.n_hid)
        self.fc3_pri = nn.Linear(
            self.n_hid * self.n_all_agents,
            self.n_hid * self.n_all_agents,
        )
        self.bn_pri = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        # Decoder
        self.n_input_dec = (
            (self.n_hid + int(self.rnn_dim / self.n_all_agents))
            * self.n_layers
            )
        self.fc1_dec = nn.Linear(self.n_input_dec, self.n_hid)
        self.fc2_dec = nn.Linear(self.n_hid, self.n_hid)
        self.fc3_dec = nn.Linear(
            self.n_hid * self.n_all_agents,
            self.n_output * self.n_players
            )
        self.bn_dec = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

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
        elif module == "dec":
            x = self.bn_dec(x)
        x = x.reshape(x.size(0), self.n_all_agents, -1)
        return x

    def Prior(self, x, h):
        _batchsize = x.size(1)
        x = torch.cat([x, h], dim=3).reshape(_batchsize, self.n_all_agents, -1)
        x = self.fc1_pri(x)
        x = F.elu(x)
        x = self.batch_norm(x, "pri")
        x = self.fc2_pri(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.reshape(_batchsize, -1)
        x = self.fc3_pri(x).reshape(_batchsize, self.n_all_agents, -1)

        del h

        return x

    def Decoder(self, z, h):
        _batchsize = z.size(1)
        x = torch.cat([z, h], dim=3).reshape(_batchsize, self.n_all_agents, -1)
        x = self.fc1_dec(x)
        x = F.elu(x)
        x = self.batch_norm(x, "dec")
        x = self.fc2_dec(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = x.reshape(_batchsize, -1)
        x = self.fc3_dec(x).reshape(_batchsize, self.n_players, -1)

        # mean = self.dec_mean(x)
        # if not self.fixedsigma:
        #     std = self.dec_std(x)

        # del x, z, h
        del z, h

        # return mean, std
        return x

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

        # Compute time differences if use_dt is enabled
        if self.use_dt:
            # Get frame numbers and compute dt
            # frames shape: (batch, seq_len, 1) or similar
            if "frames" in items:
                frames = items["frames"]  # (batch, seq_len, 1)
                # Compute time difference between consecutive frames
                # dt[t] = (frame[t] - frame[t-1]) / FRAME_RATE
                # For t=0, use default dt
                dt_values = torch.ones(seq_len, batchSize, 1).to(self.device) * DT
                for t in range(1, seq_len):
                    frame_diff = (frames[:, t, 0] - frames[:, t-1, 0]).float()
                    dt_values[t, :, 0] = frame_diff / FRAME_RATE
            else:
                # Use default dt if frames not available
                dt_values = torch.ones(seq_len, batchSize, 1).to(self.device) * DT

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

            # Training
            h_t_1 = h_t.reshape(h_t.size(0), batchSize, self.n_all_agents, -1)
            x_t_tmp = (
                x_t
                .unsqueeze(0)
                .expand(h_t_1.size(0), -1, -1, -1)
                )
            # Prior
            # Input: x^t, h^(t-1)
            z_t = self.Prior(x_t_tmp, h_t_1)
            z_t_tmp = (
                z_t
                .unsqueeze(0)
                .expand(h_t_1.size(0), -1, -1, -1)
                )

            # Decoder
            # Input: z^t, h^(t-1)
            # mean_dec, _ = self.Decoder(z_t_tmp, h_t_1)
            v_t_pred = self.Decoder(z_t_tmp, h_t_1)

            # Loss
            # Reconstruction
            player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
            player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items, t, seq_len, player_vel_GT)
            batch_losses["L_rec"] += loss_reconst(
                player_vel_pred,
                player_vel_GT,
                self.n_players,
                Sum=True
            )
            if acc == 0:
                batch_losses2["L_rec2"][t] += loss_reconst(
                    player_vel_pred,
                    player_vel_GT,
                    self.n_players,
                    Sum=False
                )

            # RNN
            # h^t = f(x^t, z^t, h^(t-1), [dt])
            rnn_input = torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(h_t.size(0), batchSize, -1)

            # Add dt as input feature if use_dt is enabled
            if self.use_dt:
                dt_t = dt_values[t].unsqueeze(0).expand(h_t.size(0), -1, -1)
                rnn_input = torch.cat([rnn_input, dt_t], dim=2)

            _, h_t = self.rnn(rnn_input, h_t)

            del (
                x_t,
                v_t,
                x_t_tmp,
                h_t_1,
                z_t,
                z_t_tmp,
                v_t_pred,
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

        # Compute time differences if use_dt is enabled
        if self.use_dt:
            if "frames" in items:
                frames = items["frames"]
                dt_values = torch.ones(seq_len, batchSize, 1).to(self.device) * DT
                for t in range(1, seq_len):
                    frame_diff = (frames[:, t, 0] - frames[:, t-1, 0]).float()
                    dt_values[t, :, 0] = frame_diff / FRAME_RATE
            else:
                dt_values = torch.ones(seq_len, batchSize, 1).to(self.device) * DT

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

                # Validation and Test
                h_t_1 = h_t.reshape(h_t.size(0), batchSize, self.n_all_agents, -1)
                x_t_tmp = (
                    x_t
                    .unsqueeze(0)
                    .expand(h_t_1.size(0), -1, -1, -1)
                    )
                # Prior
                # Input: x^t, h^(t-1)
                z_t = self.Prior(x_t_tmp, h_t_1)
                z_t_tmp = (
                    z_t
                    .unsqueeze(0)
                    .expand(h_t_1.size(0), -1, -1, -1)
                    )

                if t < seq_len - 1:
                    # RNN
                    # h^t = f(x^t, z^t, h^(t-1), [dt])
                    rnn_input = torch.cat([x_t_tmp, z_t_tmp], dim=3).reshape(h_t.size(0), batchSize, -1)

                    # Add dt as input feature if use_dt is enabled
                    if self.use_dt:
                        dt_t = dt_values[t].unsqueeze(0).expand(h_t.size(0), -1, -1)
                        rnn_input = torch.cat([rnn_input, dt_t], dim=2)

                    _, h_t = self.rnn(rnn_input, h_t)
                    del (
                        x_t,
                        v_t,
                        x_t_tmp,
                        h_t_1,
                        z_t,
                        z_t_tmp,
                    )

                elif t == seq_len - 1:
                    # Decoder
                    # Input: z^t, h^(t-1)
                    # mean_dec, _ = self.Decoder(z_t_tmp, h_t_1)
                    v_t_pred = self.Decoder(z_t_tmp, h_t_1)

                    # Loss
                    # Reconstruction
                    player_vel_GT = self.restore(v_t[:,1:,:], players_info)
                    player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
                    batch_losses["L_rec"][i] += loss_reconst(
                        player_vel_pred,
                        player_vel_GT,
                        self.n_players,
                        Sum=True
                    )
                    if acc == 0:
                        batch_losses2["L_rec2"][i] += loss_reconst(
                            player_vel_pred,
                            player_vel_GT,
                            self.n_players,
                            Sum=False
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
                        x_t_tmp,
                        h_t,
                        h_t_1,
                        z_t,
                        z_t_tmp,
                        v_t_pred,
                        player_vel_GT,
                        player_vel_pred,
                        pos_GT,
                    )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
