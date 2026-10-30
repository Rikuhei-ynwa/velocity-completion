import numpy as np
import torch

# Models that process single timestep data
MODELS_SINGLE = [
    "VGNN_add",
    "GNN_add",
    "VGNN",
    "VAE",
    "GNN",
    "MLP",
    "SVM",
    "RF",
    "GAT",
    "CrossModalGating",
    "CrossModalGating_add",  # + event type and end-ball position in the context branch
]

# Models that process sequence (multi-timestep) data
MODELS_SEQ = [
    "GVRNN_add",
    "GRNN_add",
    "GVRNN",
    "VRNN",
    "GRNN",
    "RNN",
    "GATRNN",
    "Transformer",
    "Transformer_add",
    "Transformer_add_assoc",  # + displacement to the previous event (players associated by position) and the time gap
    "Transformer_add_resid",  # same inputs; predicts the residual of the position-difference velocity
    "Transformer_add_eq",  # the three variants with one output head shared by all player slots
    "Transformer_add_assoc_eq",
    "Transformer_add_resid_eq",
    "Transformer_add_eq2",  # fully permutation-equivariant (inside a team); see models/transformer.py
    "Transformer_add_assoc_eq2",
    "Transformer_add_resid_eq2",
    "Transformer_add_flat",  # controls: same number of parameters as the eq model but a slot-specific head
    "Transformer_add_eq2_wide",  # controls: the eq2 model with as many parameters as the original model
    "TGAT",
]
# models that also get the association features of preprocess_data/make_assoc_features.py
ASSOC_MODELS = ("Transformer_add_assoc", "Transformer_add_resid", "Transformer_add_assoc_eq", "Transformer_add_resid_eq",
                "Transformer_add_assoc_eq2", "Transformer_add_resid_eq2")


# "<name>_eq": the same model with permutation-equivariant layers (models/equivariant.py), same inputs
_EQ_BASES = ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN",
             "GVRNN_add", "GAT", "CrossModalGating", "CrossModalGating_add", "GATRNN", "TGAT"]
MODELS_SINGLE += [m + sfx for m in list(MODELS_SINGLE) for sfx in ("_eq", "_eqw", "_resid_eq2") if m in _EQ_BASES]
MODELS_SEQ += [m + sfx for m in list(MODELS_SEQ) for sfx in ("_eq", "_eqw", "_resid_eq2") if m in _EQ_BASES]

class EventDataset:
    """
    EventDataset

    """

    def __init__(self, args, train=0):
        self.train = train
        self.data_path = args.data_path
        self.batchsize = args.batchsize
        self.model = args.model
        self.n_all_agents = args.n_all_agents
        self.data = self.load(args)
        self.cies = torch.tensor(self.data["cies"].astype(np.float32))
        self.players_vel_info = torch.tensor(
            self.data["vel_stats"][1, :].astype(np.float32)
        )
        self.ball_vel_info = torch.tensor(
            self.data["vel_stats"][0, :].astype(np.float32)
        )

        if "G" in self.model:
            self.node_type = torch.tensor(self.data["node_type"].astype(np.float32))
            self.edge_type = torch.tensor(self.data["edge_type"].astype(np.float32))
            if "add" in self.model and "RNN" in self.model:
                self.eventtype_data = torch.tensor(
                    self.data["eventtypes"].astype(np.float32)
                ).unsqueeze(3)
            elif "add" in self.model:
                self.eventtype_data = torch.tensor(
                    self.data["eventtypes"].astype(np.float32)
                ).unsqueeze(2)
        elif self.model.startswith("Transformer_add"):  # no graph inputs, but the event-type feature (sequence layout)
            self.eventtype_data = torch.tensor(
                self.data["eventtypes"].astype(np.float32)
            ).unsqueeze(3)
        if self.model in ASSOC_MODELS or self.model.endswith("_resid_eq2"):
            suffix = {1: "training", 0: "validation", -1: "test"}[self.train]
            datatype = "non_recurrent" if self.model in MODELS_SINGLE else "recurrent"
            z = np.load(f"{self.data_path}/dataset/{args.date_preprocessed_data}/rectangular/{datatype}/{suffix}_assoc.npz")
            self.assoc = torch.tensor(z["assoc"].astype(np.float32))  # (N, 1, 23, 5)
            assert len(self.assoc) == len(self.cies), "association features do not match the dataset"

        if self.train == -1:
            self.frame = torch.tensor(self.data["frames"])

    def load(self, args):
        """
        Parameters
        ----------
        self :
            EventDataset instance
        args :
            command line arguments

        Returns
        -------
        data :
            data of train/validation/test dataset
        """
        if self.train == 1:
            suffix = "training"
        elif self.train == 0:
            suffix = "validation"
        elif self.train == -1:
            suffix = "test"

        if self.model in MODELS_SINGLE:
            datatype = "non_recurrent"
        elif self.model in MODELS_SEQ:
            datatype = "recurrent"

        data = np.load(
            (
                self.data_path
                + f"/dataset/{args.date_preprocessed_data}/rectangular/{datatype}/{suffix}.npz"
            ),
        )

        return data

    def __len__(self):
        return self.cies.shape[0]

    def __getitem__(self, idx):
        btch_idx = idx
        items = {}

        if self.model in MODELS_SINGLE:
            items["cies"] = self.cies[btch_idx, :, :]
            if self.model.endswith("_resid_eq2"):
                items["assoc"] = self.assoc[btch_idx][0]  # (23, 5) -> (B, 23, 5)
            if "G" in self.model:
                items["node_type"] = self.node_type[btch_idx, :, :]
                items["edge_type"] = self.edge_type[btch_idx, :, :]
                if "add" in self.model:
                    items["eventtypes"] = self.eventtype_data[btch_idx, :]
            if self.train == -1:
                items["frames"] = self.frame[btch_idx, :]

        elif self.model in MODELS_SEQ:
            items["cies"] = self.cies[btch_idx, :, :, :]
            if self.model.startswith("Transformer_add"):
                items["eventtypes"] = self.eventtype_data[btch_idx, :, :, :]
            if self.model in ASSOC_MODELS or self.model.endswith("_resid_eq2"):
                items["assoc"] = self.assoc[btch_idx]  # (1, 23, 5): permuted to (1, B, 23, 5) with the other items
            if "G" in self.model:
                items["node_type"] = self.node_type[btch_idx, :, :, :]
                items["edge_type"] = self.edge_type[btch_idx, :, :, :]
                if "add" in self.model:
                    items["eventtypes"] = self.eventtype_data[btch_idx, :, :, :]
            if self.train == -1:
                items["frames"] = self.frame[btch_idx, :, :]

        return items
