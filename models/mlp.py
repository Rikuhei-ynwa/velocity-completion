import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg

from models.utils import add_difference, loss_reconst, angular_error, axis_error

# Rikuhei Umemoto, 2024
# anonymous, 2021
# modifying the code
# https://github.com/ezhan94/multiagent-programmatic-supervision
# https://github.com/keisuke198619/C-OBSO

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class MLP(nn.Module):
    """Three-layer fully-connected ELU net with batch norm."""

    def __init__(self, params, parser=None):
        super(MLP, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.params = params
        self.batchsize = params["batchsize"]
        self.n_all_agents = params["n_all_agents"]
        self.n_players = params["n_all_agents"] - 1
        self.n_input = params["n_input"]
        self.n_hid = params["n_hid"]
        self.n_output = params["n_output"]
        self.dropout_prob = params["do_prob"]

        self.fc1 = nn.Linear(self.n_input, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(
            self.n_hid * self.n_all_agents, self.n_output * self.n_players
        )
        self.bn = nn.BatchNorm1d(self.n_hid * self.n_all_agents)

        self.init_weights()

    def init_weights(self):
        for model in self.modules():
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight.data)
                if model.bias is not None:
                    model.bias.data.fill_(0.1)
            elif isinstance(model, nn.BatchNorm1d):
                model.weight.data.fill_(1)
                model.bias.data.zero_()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def batch_norm(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.bn(x)
        x = x.reshape(x.size(0), self.n_all_agents, -1)
        return x

    def restore(self, data, info):
        """
        players_vel_info = np.array(
            [players_mean_v0, players_std_v0, players_mean_v1, players_std_v1],
            )
        """
        data[:, :, 0] = data[:, :, 0] * info[1] + info[0]
        data[:, :, 1] = data[:, :, 1] * info[3] + info[2]

        return data

    def module(self, x):
        _batchsize = x.size(0)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob)
        x = self.fc2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob)
        x = x.reshape(_batchsize, -1)
        x = self.fc3(x).reshape(_batchsize, self.n_players, 2)

        return x

    def forward(self, items, players_info):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batchSize = items["cies"].size(0)

        # The dictionary for storing loss per batch is batch_losses
        # The dictionary for storing loss per event is batch_losses2
        batch_losses = {}
        batch_losses2 = {}
        batch_losses["L_kl"] = torch.zeros(1).to(self.device)
        batch_losses["L_rec"] = torch.zeros(1).to(self.device)
        if acc == 0:
            batch_losses2["L_rec2"] = torch.zeros(batchSize).to(self.device)

        x_t = torch.cat(
            [
                items["cies"][:, 0:1, :],
                items["cies"][:, 2 : len(LOC_NODES), :],
            ],
            dim=1,
        ).to(self.device)
        v_t = items["cies"][:, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :].to(
            self.device
        )

        # Training
        v_t_pred = self.module(x_t)

        # Loss
        # Reconstruction
        player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
        player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
        batch_losses["L_rec"] += loss_reconst(
            player_vel_pred,
            player_vel_GT,
            self.n_players,
            Sum=True,
        )
        if acc == 0:
            batch_losses2["L_rec2"] += loss_reconst(
                player_vel_pred,
                player_vel_GT,
                self.n_players,
                Sum=False,
            )

        del (
            x_t,
            v_t,
            v_t_pred,
            player_vel_GT,
            player_vel_pred,
        )

        return batch_losses, batch_losses2

    def sample(self, items, players_info, n_sample=1, TEST=False):
        acc = self.params["acc"]
        if acc != 0:
            raise ValueError("acc must be 0")
        batchSize = items["cies"].size(0)
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
            x_t = torch.cat(
                [cies_n[i][:, 0:1, :], cies_n[i][:, 2 : len(LOC_NODES), :]],
                dim=1,
            ).to(self.device)
            v_t = cies_n[i][:, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :].to(
                self.device
            )

            # Validation and Test
            v_t_pred = self.module(x_t)

            # Loss
            # Reconstruction
            player_vel_GT = self.restore(v_t[:, 1:, :], players_info)
            player_vel_pred = add_difference(self, self.restore(v_t_pred, players_info), items)
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

            # postition
            pos_GT = x_t.detach().cpu().numpy()
            max1 = cfg.FIELD_LENGTH / 2
            max2 = cfg.FIELD_WIDTH / 2
            pos_GT[:, :, 0] = pos_GT[:, :, 0] * max1
            pos_GT[:, :, 1] = pos_GT[:, :, 1] * max2

            if TEST:
                ball_vel_rb = np.expand_dims(
                    v_t[:, 0, :].detach().cpu().numpy(), axis=1
                )
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
                    items["frames"][:, :].detach().cpu().numpy().T[0]
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
                v_t_pred,
                player_vel_GT,
                player_vel_pred,
                pos_GT,
            )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
