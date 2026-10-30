import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import event_config as cfg
from .mlp import MLP
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


class GNN(MLP):
    def __init__(self, params, parser=None):
        super(GNN, self).__init__(params, parser=None)
        # self.on_the_ball_node = params["on_the_ball_node"]
        # self.goalkeepers = params["goalkeepers"]
        # self.attacking_dirs = params["attacking_dirs"]
        # self.on_the_ball_edge = params["on_the_ball_edge"]

        self.n_edge_type = params["n_edge_type"]
        self.n_node_type = params["n_node_type"]

        self.n_input_add = 0
        self.eventtypes = params["eventtypes"]
        self.ballend = params["ballend"]
        if self.eventtypes:
            self.n_input_add = self.n_input_add + 1
        if self.ballend:
            self.n_input_add = self.n_input_add + 1

        # Relations for the graph
        self.rel_receiver, self.rel_sender = self.setting_relations()

        self.fc1 = nn.Linear(self.n_input + self.n_input_add, self.n_hid)
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

    def setting_relations(self):
        """
        directed edges
        ball -> Home_1, Home_2, ..., Home_11, Away_1, Away_2, ..., Away_11 (22 edges)
        Home_1 -> ball, Home_2, ..., Home_11, Away_1, Away_2, ..., Away_11 (22 edges)
        ...
        Home_11 -> ball, Home_1, ..., Home_10, Away_1, Away_2, ..., Away_11 (22 edges)
        Away_1 -> ball, Home_1, Home_2, ..., Home_11, Away_2, ..., Away_11 (22 edges)
        ...
        Away_11 -> ball, Home_1, Home_2, ..., Home_11, Away_1, ..., Away_10 (22 edges)
        """
        off_diag = np.ones([self.n_all_agents, self.n_all_agents]) - np.eye(
            self.n_all_agents
        )

        rel_receiver = np.array(
            self.encode_onehot(np.where(off_diag)[0]),
        )
        rel_sender = np.array(
            self.encode_onehot(np.where(off_diag)[1]),
        )

        del off_diag

        rel_receiver = torch.FloatTensor(rel_receiver).to(self.device)
        rel_sender = torch.FloatTensor(rel_sender).to(self.device)

        return rel_receiver, rel_sender

    def encode_onehot(self, labels):
        """
        Parameters
        ----------
        labels : list
            List of labels

        Returns
        -------
        onehot : np.array
            Array of one-hot vectors
        """
        classes = set(labels)
        classes_dict = {
            c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
        }
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def edge2node(self, x, rel_receiver, node_type):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch,agents*(agents-1),hidden)
        rel_receiver : torch.Tensor
            Receiver relation tensor (agents*(agents-1),agents)
        node_type : torch.Tensor
            Node type tensor (batch,agents,n_node_type)

        Returns
        -------
        incoming : torch.Tensor
            Incoming tensor
        """
        try:
            incoming = torch.matmul(rel_receiver.t(), x)
            incoming = torch.cat([incoming, node_type], dim=2)
            del rel_receiver, node_type
        except:
            import pdb

            pdb.set_trace()
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_receiver, rel_sender, edge_type):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch,agents,hidden)
        rel_receiver : torch.Tensor
            Receiver relation tensor (agents*(agents-1),agents)
        rel_sender : torch.Tensor
            Sender relation tensor (agents*(agents-1),agents)
        edge_type : torch.Tensor
            Edge type tensor (batch,agents*(agents-1),n_edge_type)

        Returns
        -------
        edges : torch.Tensor
            Edge tensor
        """
        try:
            receivers = torch.matmul(
                rel_receiver, x
            )  # (batch,agents*(agents-1),hidden)
            senders = torch.matmul(rel_sender, x)  # (batch,agents*(agents-1),hidden)
            edges = torch.cat([senders, receivers, edge_type], dim=2)
            del receivers, senders, edge_type
        except:
            import pdb

            pdb.set_trace()
        return edges
    
    def module(self, x, edge_type, node_type):
        _batchsize = x.size(0)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.batch_norm(x)
        x = self.node2edge(x, self.rel_receiver, self.rel_sender, edge_type)
        x = self.fc2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.edge2node(x, self.rel_receiver, node_type)
        x = x.reshape(_batchsize, -1)
        x = self.fc3(x).reshape(_batchsize, self.n_players, 2)

        del edge_type, node_type

        return x
    
    def feat_ballend(self, ball_end_loc):
        _batchsize = ball_end_loc.size(0)
        ball_end_loc = (
            ball_end_loc
            .reshape(_batchsize, -1)
            .to(self.device)
            .clone()
            )
        feat_ball_end = (
            self.fc_ballend(ball_end_loc)
            .reshape(_batchsize, self.n_all_agents, -1)
            )
        del ball_end_loc
        return feat_ball_end

    def feat_eventtype(self, event_type):
        _batchsize = event_type.size(0)
        event_type = (
            event_type
            .reshape(_batchsize, -1)
            .to(self.device)
            .clone()
            )
        feat_event_type = (
            self.fc_eventtype(event_type)
            .reshape(_batchsize, self.n_all_agents, -1)
            )
        del event_type
        return feat_event_type

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
        v_t = items["cies"][
            :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
        ].to(self.device)
        edge_type = items["edge_type"][:, :, :].to(self.device)
        node_type = items["node_type"][:, :, :].to(self.device)
        if self.ballend:
            feat_ball_end = self.feat_ballend(
                items["cies"][:, 1:2, :]
                )
            x_t = torch.cat([x_t, feat_ball_end], dim=2)
            del feat_ball_end
        if self.eventtypes:
            feat_event_type = self.feat_eventtype(
                items["eventtypes"][:, :]
                )
            x_t = torch.cat([x_t, feat_event_type], dim=2)
            del feat_event_type

        # Training
        v_t_pred = self.module(x_t, edge_type, node_type)

        # Loss
        # Reconstruction
        player_vel_GT = self.restore(v_t[:, 1:, :], players_info).to(self.device)
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
            edge_type,
            node_type,
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
                [
                    cies_n[i][:, 0:1, :],
                    cies_n[i][:, 2 : len(LOC_NODES), :]
                ],
                dim=1,
            ).to(self.device)
            x_t_tmp = x_t.clone()
            v_t = cies_n[i][
                :, len(LOC_NODES) : len(LOC_NODES) + len(VEL_NODES), :
            ].to(self.device)
            edge_type = items["edge_type"][:, :, :].to(self.device)
            node_type = items["node_type"][:, :, :].to(self.device)
            if self.ballend:
                feat_ball_end = self.feat_ballend(items["cies"][:, 1:2, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_ball_end], dim=2)
                del feat_ball_end
            if self.eventtypes:
                feat_event_type = self.feat_eventtype(items["eventtypes"][:, :])
                x_t_tmp = torch.cat([x_t_tmp, feat_event_type], dim=2)
                del feat_event_type

            # Validation and Test
            v_t_pred = self.module(x_t_tmp, edge_type, node_type)

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
                    :, len(LOC_NODES + VEL_NODES) :, :
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
                        items["frames"][:,:].detach().cpu().numpy().T[0]
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
                player_vel_GT,
                player_vel_pred,
                pos_GT,
            )

        if TEST:
            return batch_losses, batch_losses2, data_test
        else:
            return batch_losses, batch_losses2
