import os, pdb, pickle, time, argparse, random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from models.utils import num_trainable_params, set_high_speed_weight

# from datasets import EventDataset, EventDataset_RNN
from datasets import EventDataset
from datasets.augment import permute_slots, SubsetWithAttrs, fast_event_weights
import event_config as cfg

# Rikuhei Umemoto, 2024
# modifying the following codes (MIT License; see THIRD_PARTY_NOTICES.md)
# https://github.com/ezhan94/multiagent-programmatic-supervision
# https://github.com/keisuke198619/C-OBSO


NUM_FIELD_PLAYERS = 22
NUM_BALL = 1

NUM_DIM_LOCATIONS = 2
NUM_DIM_VELOCITIES = 2

LOC_NODES = [
    "ball_start",
    "ball_end",
    "Home_1",
    "Home_2",
    "Home_3",
    "Home_4",
    "Home_5",
    "Home_6",
    "Home_7",
    "Home_8",
    "Home_9",
    "Home_10",
    "Home_11",
    "Away_1",
    "Away_2",
    "Away_3",
    "Away_4",
    "Away_5",
    "Away_6",
    "Away_7",
    "Away_8",
    "Away_9",
    "Away_10",
    "Away_11",
]
VEL_NODES = [
    "ball",
    "Home_1",
    "Home_2",
    "Home_3",
    "Home_4",
    "Home_5",
    "Home_6",
    "Home_7",
    "Home_8",
    "Home_9",
    "Home_10",
    "Home_11",
    "Away_1",
    "Away_2",
    "Away_3",
    "Away_4",
    "Away_5",
    "Away_6",
    "Away_7",
    "Away_8",
    "Away_9",
    "Away_10",
    "Away_11",
]
VEL_FAKE_NODES = [
    "Home_1",
    "Home_2",
    "Home_3",
    "Home_4",
    "Home_5",
    "Home_6",
    "Home_7",
    "Home_8",
    "Home_9",
    "Home_10",
    "Home_11",
    "Away_1",
    "Away_2",
    "Away_3",
    "Away_4",
    "Away_5",
    "Away_6",
    "Away_7",
    "Away_8",
    "Away_9",
    "Away_10",
    "Away_11",
]

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
    "Transformer_add_assoc",
    "Transformer_add_resid",
    "Transformer_add_eq",
    "Transformer_add_assoc_eq",
    "Transformer_add_resid_eq",
    "Transformer_add_eq2",
    "Transformer_add_assoc_eq2",
    "Transformer_add_resid_eq2",
    "Transformer_add_flat",
    "Transformer_add_eq2_wide",
    "TGAT",
]

MODELS_ADD = [
    "GNN_add",
    "CrossModalGating_add",
    "VGNN_add",
    "GRNN_add",
    "GVRNN_add",
    "Transformer_add",
    "Transformer_add_assoc",
    "Transformer_add_resid",
    "Transformer_add_eq",
    "Transformer_add_assoc_eq",
    "Transformer_add_resid_eq",
    "Transformer_add_eq2",
    "Transformer_add_assoc_eq2",
    "Transformer_add_resid_eq2",
    "Transformer_add_flat",
    "Transformer_add_eq2_wide",
]


# "<name>_eq": the same model with permutation-equivariant layers (models/equivariant.py), same inputs
_EQ_BASES = ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN",
             "GVRNN_add", "GAT", "CrossModalGating", "CrossModalGating_add", "GATRNN", "TGAT"]
MODELS_SINGLE += [m + sfx for m in list(MODELS_SINGLE) for sfx in ("_eq", "_eqw", "_resid_eq2") if m in _EQ_BASES]
MODELS_SEQ += [m + sfx for m in list(MODELS_SEQ) for sfx in ("_eq", "_eqw", "_resid_eq2") if m in _EQ_BASES]
MODELS_ADD += [m + sfx for m in list(MODELS_ADD) for sfx in ("_eq", "_eqw", "_resid_eq2") if m in _EQ_BASES]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--data", type=str, default="events",
        help="label of the data set; part of the run directory name (<model>_<data>_<seed>_rectangular_<batchsize>)",
    )
    parser.add_argument(
        "--data_root", type=str, default=os.environ.get("VC_DATA_ROOT", "data"),
        help="directory that contains dataset/<date_preprocessed_data>/rectangular/{recurrent,non_recurrent}/ "
        "(docs/DATA_SCHEMA.md); default: $VC_DATA_ROOT or ./data",
    )
    parser.add_argument(
        "--results_root", type=str, default=os.environ.get("VC_RESULTS_ROOT", "results"),
        help="trained models and test outputs are written to <results_root>/weights/; default: $VC_RESULTS_ROOT or ./results",
    )
    # parser.add_argument('--project_name', type=str, required=True, default='velocity-completion')
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--n_epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True, default=0)
    parser.add_argument(
        "--normalize", action="store_true"
    )  # the data sets are standardized when they are built
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--Beta", type=float, default=0.1)
    parser.add_argument("--start_lr", type=float, default=1e-4)
    parser.add_argument("--do_prob", type=float, default=0.0)
    parser.add_argument("--early_stop_loss", type=float, default=0.001)
    parser.add_argument("--early_stop_epoch", type=int, default=20)
    parser.add_argument("--input_acc", type=int, default=0)
    parser.add_argument("--input_vel", type=int, default=1)
    parser.add_argument("--cont", action="store_true")
    parser.add_argument("--numProcess", type=int, default=16)
    parser.add_argument("--TEST", action="store_true")
    parser.add_argument(
        "--train_frac", type=float, default=1.0,
        help="use a random fraction of the training windows (learning-curve experiment); validation and test are unchanged",
    )
    parser.add_argument(
        "--perm_augment", action="store_true",
        help="relabel the player slots inside each team at random in every training batch (data augmentation by the "
        "symmetry group of the anonymized data); models without graph inputs only",
    )
    parser.add_argument(
        "--bn_eval",
        action="store_true",
        help="Call model.train() for training epochs and model.eval() (BatchNorm running statistics) for "
        "validation and TEST; test results go to experiments/sample_eval/ instead of experiments/sample/ "
        "(the original runs never call model.eval(), i.e. everything runs in train mode)",
    )
    parser.add_argument("--date_preprocessed_data", type=str, required=True)
    parser.add_argument("--date_experiment", type=int, required=True, default=1)
    parser.add_argument(
        "--date_trained_model",
        type=int,
        default=None,
        help="Date of the trained model to load. If not specified, uses date_experiment.",
    )
    parser.add_argument(
        "--use_different_test_seeds",
        action="store_true",
        help="If set, use different seeds (0 to n_sample-1) for each test run. Otherwise, use fixed seed for all runs.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10,
        help="Number of test sample runs. Default is 10.",
    )
    parser.add_argument(
        "--use_dt",
        action="store_true",
        help="If set, use time difference (dt) as input feature for RNN models.",
    )
    parser.add_argument(
        "--hs_alpha",
        type=float,
        default=1.0,
        help="Weight on squared errors of players faster than --hs_threshold in the "
        "optimisation loss (1.0 = unweighted). Model selection and metrics stay unweighted.",
    )
    parser.add_argument(
        "--hs_oversample",
        type=float,
        default=1.0,
        help="Data augmentation for high speeds: training events with a player faster than --hs_threshold at the target event are "
        "drawn this many times as often (1.0 = off). Model selection and metrics stay unchanged.",
    )
    parser.add_argument(
        "--hs_threshold",
        type=float,
        default=5.5,
        help="Speed threshold (m/s) for --hs_alpha.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log learning curves to Weights & Biases during training "
        "(set WANDB_MODE=offline on nodes without internet, then `wandb sync`).",
    )
    parser.add_argument("--wandb_project", type=str, default="velocity-completion")
    parser.add_argument(
        "--wandb_group", type=str, default=None, help="Default: date_experiment."
    )
    args, _ = parser.parse_known_args()

    return args, parser


def add_experiment_condition(args, parser, data_path):
    """
    Parameters
    ----------
    args : argparse.Namespace
        Arguments.
    parser : argparse.ArgumentParser
        Argument parser.
    data_path : str
        Path to the data.

    Returns
    -------
    args : argparse.Namespace
        Arguments.
    """

    # args.in_sma = True # small multi-agent data
    # input 1: vel 2: vel+acc
    # output: 0: vel, 1: pos+vel, 2:vel+acc, 3: pos+vel+acc
    if args.input_vel != 1 or args.input_acc != 0:
        raise ValueError(
            "Check your data whether it includes the ball/players acceleration or not"
        )
    elif args.input_vel == 1 and args.input_acc == 0:
        args.output_dim = NUM_DIM_LOCATIONS + NUM_DIM_VELOCITIES
        args.n_feature = 4  # x, y, vx, vy
    if not torch.cuda.is_available():
        args.cuda = False
        print("cuda is not used")
    elif args.model in ["SVM", "RF"]:
        args.cuda = False
        print("cuda is not used")
    else:
        args.cuda = True
        print("cuda is used")
    args.filter = True
    args.dataset = args.data
    args.data_path = data_path
    args.clip = True  # gradient clipping
    args.save_every = 10
    args.n_all_agents = NUM_FIELD_PLAYERS + NUM_BALL
    args.sequence_length = 10
    args.ball_dim = args.n_feature
    args.start_lr = args.start_lr
    args.seed = args.seed
    args.min_lr = 1e-5
    args.n_hid = 8
    args.n_output = 2
    args.do_prob = args.do_prob
    args.validation_step = 10
    args.early_stop_loss = args.early_stop_loss
    args.early_stop_epoch = args.early_stop_epoch
    args.Beta = args.Beta

    # graph features
    if not "G" in args.model:
        args.n_output_ballend = args.n_hid
        args.n_output_eventtype = args.n_hid
    else:
        ## node features:
        # 0: Ball, 1: Players, 2: on_the_ball, 3: goalkeepers, 4: attacking_dirs
        n_node_type = cfg.NUM_NODE_FEATURES
        n_edge_type = cfg.NUM_EDGE_FEATURES
        args.n_edge_type = n_edge_type
        args.n_node_type = n_node_type
    if args.model in MODELS_SEQ:
        args.rnn_dim = args.n_all_agents * 2
        args.n_layers = 1
    if args.model.startswith("Transformer"):
        args.embed_dim = args.n_all_agents * 2
    if "add" in args.model:
        args.eventtypes = True
        args.ballend = True
    else:
        args.eventtypes = False
        args.ballend = False

    return args


def save_path_name(args, parser):
    """
    Parameters
    ----------
    args : argparse.Namespace
        Arguments.
    parser : argparse.ArgumentParser
        Argument parser.

    Returns
    -------
    model_pathname : str
        Path to load the trained model from.
    result_pathname : str
        Path to save the test results.
    experiment_cond : str
        Experiment condition string for saving results.
    """
    # Base path for experiments
    base_path = path_weight
    base_path = base_path + "vel_" + str(args.input_vel) + "_"
    base_path = base_path + "acc_" + str(args.input_acc) + "_"
    base_path = base_path + "norm/" if args.normalize else base_path + "unnorm/"

    # Model suffix (common for both paths)
    # Add _dt suffix if use_dt is enabled for RNN models
    model_name = args.model
    if hasattr(args, 'use_dt') and args.use_dt and "RNN" in args.model:
        model_name = args.model + "_dt"
    # Add _hs{alpha} (and threshold if not 5.5) for high-speed weighted losses
    if args.hs_alpha != 1.0:
        model_name += f"_hs{args.hs_alpha:g}"
        if args.hs_threshold != 5.5:
            model_name += f"t{args.hs_threshold:g}"
    if args.hs_oversample != 1.0:
        model_name += f"_os{args.hs_oversample:g}"

    model_suffix = (
        model_name
        + "_"
        + args.data
        + "_"
        + str(args.seed)
        + "_"
        + "rectangular"
        + "_"
        + str(args.batchsize)
        + "/"
    )

    # Path for loading trained model
    if args.date_trained_model is not None:
        model_date = args.date_trained_model
    elif args.date_experiment:
        model_date = args.date_experiment
    else:
        model_date = datetime.now().strftime("%Y%m%d%H%M")
    model_cond = base_path + f"{model_date}/" + model_suffix
    model_pathname = model_cond + "state_dict"

    # Path for saving results
    if args.date_experiment:
        result_date = args.date_experiment
    else:
        result_date = datetime.now().strftime("%Y%m%d%H%M")
    experiment_cond = base_path + f"{result_date}/" + model_suffix
    result_pathname = experiment_cond + "state_dict"

    print("model load path:\n", model_cond)
    print("result save path:\n", experiment_cond)
    os.makedirs(result_pathname, exist_ok=True)

    return model_pathname, result_pathname, experiment_cond


def train_epoch(
    params,
    model,
    optimizer,
    items,
    hp,
    batch_idx,
    losses,
    losses2,
):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    optimizer : torch.optim.Optimizer
        Optimizer.
    items : dict
        Items.
    hp : dict
        Hyperparameters.
    batch_idx : int
        Batch index.
    losses : dict
        Losses.
    losses2 : dict
        Losses.

    Returns
    -------
    losses : dict
        Losses.
    losses2 : dict
        Losses.
    """
    optimizer.zero_grad()
    # The dictionary for storing loss per batch is batch_losses
    # The dictionary for storing loss per event is batch_losses2
    batch_losses, batch_losses2 = model(items, params["train_players_vel_info"])
    total_loss = hp["Beta"] * batch_losses["L_kl"] + batch_losses["L_rec"]
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e10)
    optimizer.step()

    for key in batch_losses.keys():
        if batch_idx == 0:
            losses[key] = batch_losses[key].item()
        else:
            losses[key] = losses[key] + batch_losses[key].item()

    for key in batch_losses2.keys():
        if batch_idx == 0:
            losses2[key] = batch_losses2[key].detach().cpu().numpy()
        else:
            if params["model"] in MODELS_SINGLE:
                losses2_tmp = batch_losses2[key].detach().cpu().numpy()
                losses2[key] = np.concatenate([losses2[key], losses2_tmp], axis=0)
            elif params["model"] in MODELS_SEQ:
                losses2_tmp = batch_losses2[key].detach().cpu().numpy()
                losses2[key] = np.concatenate([losses2[key], losses2_tmp], axis=1)

    return losses, losses2


def val_epoch(params, model, items, batch_idx, losses, losses2, n_sample: int = 1):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    items: dict
        Items.
    batch_idx : int
        Batch index.
    losses : dict
        Losses.
    losses2 : dict
        Losses.
    n_sample : int, optional
        Number of samples.

    Returns
    -------
    losses : dict
        Losses.
    losses2 : dict
        Losses.
    """
    batch_losses, batch_losses2 = model.sample(
        items, params["val_players_vel_info"], n_sample
    )

    for key in batch_losses:
        if batch_idx == 0:
            losses[key] = torch.mean(batch_losses[key]).item()
        else:
            losses[key] += torch.mean(batch_losses[key]).item()

    for key in batch_losses2:
        if batch_idx == 0:
            losses2[key] = torch.mean(batch_losses2[key], dim=0).detach().cpu().numpy()
        else:
            losses2_tmp = torch.mean(batch_losses2[key], dim=0).detach().cpu().numpy()
            losses2[key] = np.concatenate([losses2[key], losses2_tmp], axis=0)

    return losses, losses2


def run_epoch(model, loader, params, train, hp):
    """
    Parameters
    ----------
    model : torch.nn.Module
        Model.
    loader : torch.utils.data.DataLoader
        Data loader.
    train : int
        Training flag.
    hp : dict
        Hyperparameters.

    Returns
    -------
    losses : dict
        Losses.
    """

    if args.bn_eval:
        # BatchNorm uses batch statistics only while training; validation runs with running statistics
        model.train() if train == 1 else model.eval()

    losses = {}
    losses2 = {}
    for batch_idx, items in enumerate(loader):
        d1 = {"batch_idx": batch_idx}
        hp.update(d1)
        if train == 1 and args.perm_augment:
            items = permute_slots(items, params["model"] in MODELS_SEQ)

        if params["model"] in MODELS_SEQ:
            # batchsize, sequence_length, n_all_agents, n_feature
            # -> sequence_length, batchsize, n_all_agents, n_feature
            for key in items.keys():
                items[key] = items[key].permute(1, 0, 2, 3)

        if args.cuda:
            for key in items.keys():
                items[key] = items[key].cuda()

        if train == 1:
            losses, losses2 = train_epoch(
                params,
                model,
                optimizer,
                items,
                hp,
                batch_idx,
                losses,
                losses2,
            )

        else:
            losses, losses2 = val_epoch(
                params, model, items, batch_idx, losses, losses2, n_sample=1
            )

    for key in losses:
        losses[key] /= len(loader.dataset)
    for key in losses2:
        if train == 1:
            if key == "L_rec2":
                losses2[key] = np.sum(np.sqrt(losses2[key])) / (
                    len(loader.dataset) * losses2[key].shape[0]
                )
        else:
            if key == "L_rec2":
                losses2[key] = np.sum(np.sqrt(losses2[key])) / len(loader.dataset)

    return losses, losses2


def loss_str(losses, train: int):
    ret = ""
    for key in losses:
        if train == 1 or train == 0:
            if "L_kl" in key or "L_rec" in key:
                ret += " {}: {:.5f} |".format(key, losses[key])
            elif "L_rec2" in key:
                ret += " {}: {:.5f} |".format(key, losses[key])
            elif "L_efobso" in key:
                ret += " {}: {:.5f} |".format(key, losses[key])
        elif train == -1:
            ret += " {}: {} |".format(key, losses[key])
    return ret


def bound(val, lower, upper):
    """Clamps val between lower and upper."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


if __name__ == "__main__":
    args, parser = parse_args()

    numProcess = args.numProcess
    os.environ["OMP_NUM_THREADS"] = str(numProcess)
    TEST = args.TEST

    data_path = args.data_root
    path_weight = os.path.join(args.results_root, "weights") + "/"

    # pre-process----------------------------------------------
    args = add_experiment_condition(args, parser, data_path)
    set_high_speed_weight(args.hs_alpha, args.hs_threshold)
    model_pathname, result_pathname, experiment_cond = save_path_name(args, parser)

    params = {
        "model": args.model,
        "epoch": args.n_epoch,
        "batchsize": args.batchsize,
        "generative": False,
        "lr": max(args.start_lr, args.min_lr),
        "normalize": args.normalize,
        "seed": args.seed,
        "cuda": args.cuda,
        "acc": args.input_acc,
        "n_all_agents": args.n_all_agents,
        "ball_dim": args.ball_dim,
        "Beta": args.Beta,
        "do_prob": args.do_prob,
        "n_input": NUM_DIM_LOCATIONS,
        "n_hid": args.n_hid,
        "n_output": args.n_output,
        "n_seq": args.sequence_length,
        "validation_step": args.validation_step,
        "early_stop_loss": args.early_stop_loss,
        "early_stop_epoch": args.early_stop_epoch,
        "eventtypes": args.eventtypes,
        "ballend": args.ballend,
    }

    if "G" in args.model:
        params["n_edge_type"] = args.n_edge_type
        params["n_node_type"] = args.n_node_type

    if "RNN" in args.model:
        params["rnn_dim"] = args.rnn_dim
        params["n_layers"] = args.n_layers
        params["use_dt"] = args.use_dt

    params["hs_alpha"] = args.hs_alpha
    params["hs_threshold"] = args.hs_threshold
    params["hs_oversample"] = args.hs_oversample
    # provenance of the run (the data set a run was trained on is otherwise only in the slurm log)
    params["date_preprocessed_data"] = args.date_preprocessed_data
    params["train_frac"] = args.train_frac
    params["perm_augment"] = bool(args.perm_augment)
    params["bn_eval"] = bool(getattr(args, "bn_eval", False))

    # Set manual seed
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Deterministic GPU computation: cuBLAS needs a fixed workspace (set before the first CUDA
    # call), and cuDNN autotuning (benchmark) must be off because the algorithm it picks depends
    # on timings, e.g. when a GPU is shared. warn_only reports ops without a deterministic kernel.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create save path and saving parameters
    pickle.dump(params, open(experiment_cond + "/params.p", "wb"), protocol=4)

    # Load model
    from models import load_model

    model = load_model(args.model, params, parser)
    # print("models :\n", model)

    # Continue a previous experiment, or start a new one
    if args.cont:
        print("args.cont = True")
        if os.path.exists("{}_best.pth".format(model_pathname)):
            state_dict = torch.load(
                "{}_best.pth".format(model_pathname), weights_only=True
            )
            model.load_state_dict(state_dict)
            print("best model was loaded from: {}_best.pth".format(model_pathname))
        else:
            print("args.cont = True but file did not exist")
    else:
        print("args.cont = False")

    print("############################################################")

    # Dataset loaders
    num_workers = args.numProcess
    kwargs = {"num_workers": num_workers, "pin_memory": True} if args.cuda else {}
    print("num_workers:" + str(num_workers))

    if args.model in MODELS_SINGLE:
        if not TEST:
            dataset_train = EventDataset(args, train=1)
            if args.train_frac < 1.0:
                dataset_train = SubsetWithAttrs(dataset_train, args.train_frac, args.seed)
                print(f"training windows: {len(dataset_train)} ({args.train_frac:.0%} of the training set)")
            sampler = None
            if args.hs_oversample != 1.0:
                w_train, frac_fast = fast_event_weights(dataset_train, args.hs_oversample, args.hs_threshold)
                sampler = torch.utils.data.WeightedRandomSampler(w_train, num_samples=len(dataset_train), replacement=True, generator=g)
                print(f"high-speed oversampling x{args.hs_oversample:g}: {frac_fast:.1%} of the training events have a player faster than {args.hs_threshold} m/s")
            loader_train = DataLoader(
                dataset_train,
                batch_size=args.batchsize,
                shuffle=sampler is None,
                sampler=sampler,
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs,
            )
            # statistics of the training targets
            v_t_players = []
            for items in loader_train:
                v_t = items["cies"][
                    :,
                    len(cfg.LOC_NODES) : len(cfg.LOC_NODES) + len(cfg.VEL_NODES),
                    :,
                ]
                v_t_players.append(v_t[:, 1:, :].detach().cpu().numpy())
            v_t_players = np.concatenate(v_t_players, axis=0)
            players_vel_info_train = (
                dataset_train.players_vel_info.detach().cpu().numpy()
            )
            v_t_players[:, :, 0] = (
                v_t_players[:, :, 0] * players_vel_info_train[1]
                + players_vel_info_train[0]
            )
            v_t_players[:, :, 1] = (
                v_t_players[:, :, 1] * players_vel_info_train[3]
                + players_vel_info_train[2]
            )
            experiment_path = "{}/experiments/sample".format(experiment_cond)
            os.makedirs(experiment_path, exist_ok=True)
            fig, ax = plt.subplots()
            ax.hist(
                np.sqrt(
                    v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                ).flatten(),
                bins=50,
                range=(0, cfg.PLAYERS_SPEED_LIMIT),
                alpha=0.5,
                color="blue",
                label="Ground truth",
            )
            ax.set_xlabel("Velocity (m/s)", fontsize=16)
            ax.set_ylabel("Frequency", fontsize=16)
            ax.set_ylim(0, 30000)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig("{}/train_mlp_vel_hist.png".format(experiment_path))
            print(
                "Saved the histogram of the player velocities in the training dataset."
            )
            num_data = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
            ).size
            num_high_speed = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
                > 5.5
            ).sum()
            num_high_intensity = (
                (
                    np.sqrt(
                        v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                    ).flatten()
                    > 5.5
                )
                & (
                    np.sqrt(
                        v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                    ).flatten()
                    <= 7.0
                )
            ).sum()
            num_sprint = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
                > 7.0
            ).sum()
            print("N:" + str(num_data))
            print(
                "Number of high-speed running (>5.5 m/s): "
                + str(num_high_speed)
                + " ("
                + str(round(100 * num_high_speed / num_data, 2))
                + " %)"
            )
            print(
                "Number of high-intensity running (>5.5 m/s and <=7.0 m/s): "
                + str(num_high_intensity)
                + " ("
                + str(round(100 * num_high_intensity / num_data, 2))
                + " %)"
            )
            print(
                "Number of sprints (>7.0 m/s): "
                + str(num_sprint)
                + " ("
                + str(round(100 * num_sprint / num_data, 2))
                + " %)"
            )
            plt.clf()
            plt.close()

            dataset_val = EventDataset(args, train=0)
            loader_val = DataLoader(
                dataset_val,
                batch_size=args.batchsize,
                shuffle=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs,
            )

        dataset_test = EventDataset(args, train=-1)
        loader_test = DataLoader(
            dataset_test,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs,
        )

    elif args.model in MODELS_SEQ:
        if not TEST:
            # dataset_train = EventDataset_RNN(
            dataset_train = EventDataset(args, train=1)
            if args.train_frac < 1.0:
                dataset_train = SubsetWithAttrs(dataset_train, args.train_frac, args.seed)
                print(f"training windows: {len(dataset_train)} ({args.train_frac:.0%} of the training set)")
            sampler = None
            if args.hs_oversample != 1.0:
                w_train, frac_fast = fast_event_weights(dataset_train, args.hs_oversample, args.hs_threshold)
                sampler = torch.utils.data.WeightedRandomSampler(w_train, num_samples=len(dataset_train), replacement=True, generator=g)
                print(f"high-speed oversampling x{args.hs_oversample:g}: {frac_fast:.1%} of the training events have a player faster than {args.hs_threshold} m/s")
            loader_train = DataLoader(
                dataset_train,
                batch_size=args.batchsize,
                shuffle=sampler is None,
                sampler=sampler,
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs,
            )
            # statistics of the training targets
            v_t_players = []
            for items in loader_train:
                v_t = items["cies"][
                    :,
                    items["cies"].size(1) - 1,
                    len(cfg.LOC_NODES) : len(cfg.LOC_NODES) + len(cfg.VEL_NODES),
                    :,
                ]
                v_t_players.append(v_t[:, 1:, :].detach().cpu().numpy())
            v_t_players = np.concatenate(v_t_players, axis=0)
            players_vel_info_train = (
                dataset_train.players_vel_info.detach().cpu().numpy()
            )
            v_t_players[:, :, 0] = (
                v_t_players[:, :, 0] * players_vel_info_train[1]
                + players_vel_info_train[0]
            )
            v_t_players[:, :, 1] = (
                v_t_players[:, :, 1] * players_vel_info_train[3]
                + players_vel_info_train[2]
            )
            experiment_path = "{}/experiments/sample".format(experiment_cond)
            os.makedirs(experiment_path, exist_ok=True)
            fig, ax = plt.subplots()
            num_data, bins, _ = ax.hist(
                np.sqrt(
                    v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                ).flatten(),
                bins=50,
                range=(0, cfg.PLAYERS_SPEED_LIMIT),
                alpha=0.5,
                color="blue",
                label="Ground truth",
            )
            ax.set_xlabel("Velocity (m/s)", fontsize=16)
            ax.set_ylabel("Frequency", fontsize=16)
            ax.set_ylim(0, 30000)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            ax.legend(fontsize=16)
            plt.tight_layout()
            plt.savefig("{}/train_rnn_vel_hist.png".format(experiment_path))
            print(
                "Saved the histogram of the player velocities in the training dataset."
            )
            num_data = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
            ).size
            num_high_speed = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
                > 5.5
            ).sum()
            num_high_intensity = (
                (
                    np.sqrt(
                        v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                    ).flatten()
                    > 5.5
                )
                & (
                    np.sqrt(
                        v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2
                    ).flatten()
                    <= 7.0
                )
            ).sum()
            num_sprint = (
                np.sqrt(v_t_players[:, :, 0] ** 2 + v_t_players[:, :, 1] ** 2).flatten()
                > 7.0
            ).sum()
            print("N:" + str(num_data))
            print(
                "Number of high-speed running (>5.5 m/s): "
                + str(num_high_speed)
                + " ("
                + str(round(100 * num_high_speed / num_data, 2))
                + " %)"
            )
            print(
                "Number of high-intensity running (>5.5 m/s and <=7.0 m/s): "
                + str(num_high_intensity)
                + " ("
                + str(round(100 * num_high_intensity / num_data, 2))
                + " %)"
            )
            print(
                "Number of sprints (>7.0 m/s): "
                + str(num_sprint)
                + " ("
                + str(round(100 * num_sprint / num_data, 2))
                + " %)"
            )
            plt.clf()
            plt.close()

            # dataset_val = EventDataset_RNN(
            dataset_val = EventDataset(args, train=0)
            loader_val = DataLoader(
                dataset_val,
                batch_size=args.batchsize,
                shuffle=True,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs,
            )

        # dataset_test = EventDataset_RNN(
        dataset_test = EventDataset(args, train=-1)
        loader_test = DataLoader(
            dataset_test,
            batch_size=args.batchsize,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs,
        )

    ###### TRAIN LOOP ############################################
    # Load model
    from models import load_model

    model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    # Update params with model parameters
    params = model.params
    if not args.model in ["SVM", "RF"]:
        params["total_params"] = num_trainable_params(model)
        print("Total number of parameters: {}".format(params["total_params"]))
    else:
        params["total_params"] = "N/A"
    best_val_loss = 1000000
    best_epoch = 0
    epochs_since_best = 0
    epoch_first_best = -1

    if params["model"] in MODELS_ADD:
        group_name = "add"
    else:
        group_name = "noadd"
    use_wandb = args.wandb and not TEST and args.model not in ["SVM", "RF"]
    if use_wandb:
        import wandb  # optional dependency, only for --wandb

        run_name = experiment_cond.split("vel_1_acc_0_norm/")[-1].rstrip("/")
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or str(args.date_experiment),
            name=run_name,
            job_type="train",
            tags=[str(args.date_preprocessed_data), group_name],
            config={
                **{
                    k: v
                    for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
                "date_preprocessed_data": args.date_preprocessed_data,
                "date_experiment": args.date_experiment,
                "result_dir": experiment_cond,
            },
        )

    hyperparams = {
        "model": args.model,
        "acc": args.input_acc,
        "Beta": args.Beta,
        "do_prob": args.do_prob,
        "n_hid": args.n_hid,
        "validation_step": args.validation_step,
        "early_stop_loss": args.early_stop_loss,
        "early_stop_epoch": args.early_stop_epoch,
    }

    learning_process = {}

    start_time = time.time()
    if not TEST:
        params["train_players_vel_info"] = dataset_train.players_vel_info
        params["val_players_vel_info"] = dataset_val.players_vel_info
        # For SVM and RF: fit the model with train and validation data in first epoch
        if args.model in ["SVM", "RF"] and not model.is_fitted:
            print(
                "Fitting {} model with train and validation data...".format(args.model)
            )
            val_rmse, val_score = model.fit_with_datasets(
                loader_train,
                loader_val,
                params["train_players_vel_info"],
                params["val_players_vel_info"],
            )

            # Save the fitted model immediately
            if model.is_fitted:
                filename = "{}_fitted.pkl".format(result_pathname)
                with open(filename, "wb") as f:
                    pickle.dump(model, f)
                # torch.save(model.state_dict(), filename)
                print(f"{args.model} fitted model saved to: {filename}")
        else:
            for epoch in range(1, args.n_epoch + 1, 1):
                print("epoch " + str(epoch))
                print("########## lr {} ##########".format(params["lr"]))
                epochs_since_best += 1
                print("########## dropout {} ##########".format(params["do_prob"]))
                print("########## Beta {} ##########".format(params["Beta"]))

                # Adam optimizer
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=params["lr"],
                )
                train_loss, train_loss2 = run_epoch(
                    model,
                    loader_train,
                    params,
                    train=1,
                    hp=hyperparams,
                )

                val_loss = {}
                val_loss2 = {}
                val_loss_min = {}
                val_loss2_min = {}
                for n in range(args.validation_step):
                    val_loss_tmp, val_loss2_tmp = run_epoch(
                        model, loader_val, params, train=0, hp=hyperparams
                    )
                    for key in val_loss_tmp:
                        if n == 0:
                            val_loss[key] = val_loss_tmp[key]
                            if key == "L_rec":
                                val_loss_min[key] = val_loss_tmp[key]
                        else:
                            val_loss[key] += val_loss_tmp[key]
                            if key == "L_rec":
                                val_loss_min[key] = min(
                                    val_loss_min[key], val_loss_tmp[key]
                                )
                    for key in val_loss2_tmp:
                        if n == 0:
                            val_loss2[key] = val_loss2_tmp[key]
                            if key == "L_rec2":
                                val_loss2_min[key] = val_loss2_tmp[key]
                        else:
                            val_loss2[key] += val_loss2_tmp[key]
                            if key == "L_rec2":
                                val_loss2_min[key] = min(
                                    val_loss2_min[key], val_loss2_tmp[key]
                                )
                for key in val_loss:
                    val_loss[key] /= args.validation_step
                for key in val_loss2:
                    val_loss2[key] /= args.validation_step

                # Save learning process
                learning_process[epoch] = {
                    "train_loss": train_loss,
                    "train_loss2": train_loss2,
                    "val_loss": val_loss,
                    "val_loss2": val_loss2,
                    "val_loss_min": val_loss_min,
                    "val_loss2_min": val_loss2_min,
                }
                print(loss_str(train_loss, 1) + loss_str(val_loss, 0))

                if use_wandb:
                    log = {
                        f"{split}/{key}": float(np.mean(value))
                        for split, losses_split in (
                            ("train", train_loss),
                            ("train", train_loss2),
                            ("val", val_loss),
                            ("val", val_loss2),
                        )
                        for key, value in losses_split.items()
                    }
                    log["time/elapsed_s"] = time.time() - start_time
                    wandb.log(log, step=epoch)

                # Best model on validation set
                total_val_loss = val_loss2["L_rec2"]
                if epoch > epoch_first_best and total_val_loss < best_val_loss:
                    best_val_loss_prev = best_val_loss
                    best_val_loss = total_val_loss
                    best_epoch = epoch
                    epochs_since_best = 0

                    filename = "{}_best.pth".format(result_pathname)

                    torch.save(model.state_dict(), filename)
                    print(filename + " is saved")
                    print("##### Best model #####")

                    diff_loss = best_val_loss_prev - best_val_loss
                    # if diff_loss < args.early_stop_loss:
                    #     print(
                    #         "best epoch: {} loss: {:.5f}".format(best_epoch, best_val_loss)
                    #         )
                    #     epoch_time = time.time() - start_time
                    #     print("Time:\t {:.3f}".format(epoch_time))
                    #     break

                if epochs_since_best > args.early_stop_epoch:
                    print(
                        "best validation epoch: {} loss: {:.5f}".format(
                            best_epoch, best_val_loss
                        )
                    )
                    epoch_time = time.time() - start_time
                    print("Time:\t {:.3f}".format(epoch_time))
                    break

                # Periodically save model
                if epoch % args.save_every == 0:
                    filename = "{}_epoch_{}.pth".format(result_pathname, epoch)
                    torch.save(model.state_dict(), filename)
                    print(filename + " is saved")
                    print("########## Saved model ##########")

                epoch_time = time.time() - start_time
                print("Time:\t {:.3f}".format(epoch_time))

            print(
                "Best validation epoch: {} loss: {:.4f}".format(
                    best_epoch, best_val_loss
                )
            )
            if use_wandb:
                wandb.run.summary["best_epoch"] = best_epoch
                wandb.run.summary["best_val/L_rec2"] = float(best_val_loss)
                wandb.finish()

            # Save learning process
            with open(result_pathname + "/learning_process.pkl", "wb") as f:
                pickle.dump(learning_process, f)

        print("Training is finished")

        # =====================================================================
        # Training Summary - Saved Files
        # =====================================================================
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY - SAVED FILES")
        print("=" * 70)
        print(f"Base directory: {experiment_cond}")
        print("-" * 70)
        print("Saved files:")
        print(f"  - params.p              : {experiment_cond}/params.p")
        print(f"  - Best model            : {result_pathname}_best.pth")
        if args.model in ["SVM", "RF"]:
            print(f"  - Fitted model          : {result_pathname}_fitted.pkl")
        else:
            print(f"  - Learning process      : {result_pathname}/learning_process.pkl")
        print("=" * 70 + "\n")

    # Determine model loading path (use model_pathname for loading pre-trained models)
    # If date_trained_model is specified, model_pathname points to the trained model
    # Otherwise, model_pathname == result_pathname
    model_cond = os.path.dirname(model_pathname) + "/"

    # Load params from the trained model path
    params = pickle.load(open(model_cond + "/params.p", "rb"))
    print(f"Loaded params from: {model_cond}/params.p")

    # Load model
    if args.model in ["SVM", "RF"]:
        # For SVM/RF, try to load fitted model first
        fitted_model_path = "{}_fitted.pkl".format(model_pathname)

        loaded = False
        if os.path.exists(fitted_model_path):
            try:
                with open(fitted_model_path, "rb") as f:
                    loaded_model = pickle.load(f)
                # Copy the fitted sklearn model
                model.model = loaded_model.model
                model.is_fitted = loaded_model.is_fitted
                if hasattr(loaded_model, "best_params_"):
                    model.best_params_ = loaded_model.best_params_
                print(f"Loaded {args.model} fitted model from: {fitted_model_path}")
                loaded = True
            except Exception as e:
                print(f"Failed to load fitted model: {e}")

        if not loaded:
            print(f"No fitted model found, using untrained {args.model} model")
    else:
        # For PyTorch models
        state_dict = torch.load(
            "{}_best.pth".format(model_pathname),
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        print(f"Loaded model from: {model_pathname}_best.pth")
        if args.bn_eval:
            model.eval()  # persists over the reloads of the test runs below
            print("model.eval(): BatchNorm uses running statistics")

    # Load ground-truth states from test set
    loader = loader_test
    n_sample = args.n_sample
    n_smp_b = 1
    rep_smp = int(n_sample / n_smp_b)

    if True:
        # Sample velocities
        print("test sample")
        if args.use_different_test_seeds:
            print(
                "Using different seeds for each test run (seeds: 0 to {})".format(
                    n_sample - 1
                )
            )
        else:
            print("Using fixed seed ({}) for all test runs".format(args.seed))

        # Initialize losses
        loss_i = [{} for t in range(n_sample)]
        losses = {}
        losses2 = {}
        others = {}

        for r in range(rep_smp):
            start_time = time.time()

            # Set seed for this test run
            if args.use_different_test_seeds:
                test_seed = r  # Use seeds 0, 1, 2, ..., n_sample-1
                random.seed(test_seed)
                np.random.seed(test_seed)
                torch.manual_seed(test_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(test_seed)
                print(f"Test run {r}: using seed {test_seed}")

            if r > 0:
                # Reload model for multiple runs
                if args.model in ["SVM", "RF"]:
                    # For SVM/RF, reload from pickle
                    fitted_model_path = "{}_fitted.pkl".format(model_pathname)
                    if os.path.exists(fitted_model_path):
                        try:
                            with open(fitted_model_path, "rb") as f:
                                loaded_model = pickle.load(f)
                            model.model = loaded_model.model
                            model.is_fitted = loaded_model.is_fitted
                        except:
                            print(f"Failed to reload {args.model} model")
                else:
                    # For PyTorch models
                    state_dict = torch.load(
                        "{}_best.pth".format(model_pathname),
                        map_location=lambda storage, loc: storage,
                        weights_only=True,
                    )
                    model.load_state_dict(state_dict)

            players_info = dataset_test.players_vel_info
            ball_info = dataset_test.ball_vel_info.detach().cpu().numpy()

            # For SVM/RF: Run direct test evaluation before sampling
            if args.model in ["SVM", "RF"] and model.is_fitted:
                print(f"Running direct {args.model} test evaluation...")
                # Define experiment_path for SVM/RF
                experiment_path = "{}/experiments/sample{}".format(
                    experiment_cond, "_eval" if args.bn_eval else ""
                )
                os.makedirs(experiment_path, exist_ok=True)

                if r == 0:
                    losses["L_rec2"] = np.zeros(n_sample)
                    losses2["L_efobso"] = np.zeros((n_sample, len(loader.dataset)))
                test_rmse, test_r2, l_efobsos = model.test(
                    loader_test, dataset_test.players_vel_info, n_sample=1
                )
                losses["L_rec2"][r * n_smp_b : (r + 1) * n_smp_b] = np.mean(test_rmse)
                losses2["L_efobso"][
                    r * n_smp_b : (r + 1) * n_smp_b,
                    :,
                ] += np.mean(l_efobsos, axis=0)

                for i in range(n_smp_b):
                    loss_i[r * n_smp_b + i]["L_rec2"] = losses["L_rec2"][
                        r * n_smp_b + i
                    ]
                    loss_i[r * n_smp_b + i]["L_efobso"] = np.mean(
                        np.sqrt(losses2["L_efobso"][r * n_smp_b + i])
                    )
                print(
                    "Test sample "
                    + str(r * n_smp_b + i)
                    + ":\t"
                    + loss_str(loss_i[r * n_smp_b + i], train=-1)
                )

            else:
                for batch_idx, items in enumerate(loader):
                    if params["model"] in MODELS_SEQ:
                        for key in items.keys():
                            if key == "frames":
                                continue
                            else:
                                # batchsize, sequence_length, n_all_agents, n_feature
                                # -> sequence_length, batchsize, n_all_agents, n_feature
                                items[key] = items[key].permute(1, 0, 2, 3)

                    if args.cuda:
                        for key in items.keys():
                            items[key] = items[key].cuda()

                    batch_losses, batch_losses2, data_test = model.sample(
                        items, players_info, n_sample=n_smp_b, TEST=True
                    )

                    for key in batch_losses:
                        if batch_idx == 0 and r == 0:
                            losses[key] = np.zeros(n_sample)
                        losses[key][r * n_smp_b : (r + 1) * n_smp_b] += batch_losses[
                            key
                        ].item()

                    for key in batch_losses2:
                        if batch_idx == 0 and r == 0:
                            losses2[key] = np.zeros((n_sample, len(loader.dataset)))
                        batch_loss_tmp = torch.mean(batch_losses2[key], dim=0)
                        if batch_loss_tmp.shape[0] < args.batchsize:
                            losses2[key][
                                r * n_smp_b : (r + 1) * n_smp_b,
                                -batch_loss_tmp.shape[0] :,
                            ] += (
                                batch_loss_tmp.detach().cpu().numpy()
                            )
                        else:
                            losses2[key][
                                r * n_smp_b : (r + 1) * n_smp_b,
                                batch_idx
                                * args.batchsize : (batch_idx + 1)
                                * args.batchsize,
                            ] += (
                                batch_loss_tmp.detach().cpu().numpy()
                            )

                    for key in data_test:
                        if batch_idx == 0 and r == 0:
                            if key in [
                                "L_efobso",
                                "L_angular",
                                "L_axis_x",
                                "L_axis_y",
                                "L_angular_rb",
                                "L_axis_x_rb",
                                "L_axis_y_rb",
                            ]:
                                losses2[key] = np.zeros((n_sample, len(loader.dataset)))
                            elif key == "loc_gt" or key == "v_rb":
                                others[key] = np.zeros(
                                    (
                                        n_sample,
                                        len(loader.dataset),
                                        args.n_all_agents,
                                        2,
                                    )
                                )
                            elif key == "frames":
                                others[key] = np.zeros((n_sample, len(loader.dataset)))
                            else:
                                others[key] = np.zeros(
                                    (
                                        n_sample,
                                        len(loader.dataset),
                                        args.n_all_agents - 1,
                                        2,
                                    )
                                )
                        if key in [
                            "L_efobso",
                            "L_angular",
                            "L_axis_x",
                            "L_axis_y",
                            "L_angular_rb",
                            "L_axis_x_rb",
                            "L_axis_y_rb",
                        ]:
                            if data_test[key].shape[1] < args.batchsize:
                                losses2[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    -data_test[key].shape[1] :,
                                ] += np.mean(data_test[key], axis=0)
                            else:
                                losses2[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    batch_idx
                                    * args.batchsize : (batch_idx + 1)
                                    * args.batchsize,
                                ] += np.mean(data_test[key], axis=0)
                        elif key == "v_rb":
                            v_rb = np.mean(data_test[key], axis=0)
                            # Destandarize the ball velocity
                            v_rb[:, 0, 0] = v_rb[:, 0, 0] * ball_info[1] + ball_info[0]
                            v_rb[:, 0, 1] = v_rb[:, 0, 1] * ball_info[3] + ball_info[2]
                            if data_test[key].shape[1] < args.batchsize:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    -data_test[key].shape[1] :,
                                ] += v_rb
                            else:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    batch_idx
                                    * args.batchsize : (batch_idx + 1)
                                    * args.batchsize,
                                ] += v_rb
                        elif key == "frames":
                            if data_test[key].shape[1] < args.batchsize:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    -data_test[key].shape[1] :,
                                ] += np.mean(data_test[key], axis=0)
                            else:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    batch_idx
                                    * args.batchsize : (batch_idx + 1)
                                    * args.batchsize,
                                ] += np.mean(data_test[key], axis=0)
                        else:
                            if data_test[key].shape[1] < args.batchsize:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    -data_test[key].shape[1] :,
                                ] += np.mean(data_test[key], axis=0)
                            else:
                                others[key][
                                    r * n_smp_b : (r + 1) * n_smp_b,
                                    batch_idx
                                    * args.batchsize : (batch_idx + 1)
                                    * args.batchsize,
                                ] += np.mean(data_test[key], axis=0)

                for i in range(n_smp_b):
                    for key in losses:
                        if key == "L_rec":
                            loss_i[r * n_smp_b + i][key] = losses[key][
                                r * n_smp_b + i
                            ] / len(loader.dataset)
                    for key in losses2:
                        #             for key in losses2:
                        # if train == 1:
                        #     if key == "L_rec2":
                        #         losses2[key] = np.sum(np.sqrt(losses2[key])) / (
                        #             len(loader.dataset) * losses2[key].shape[0]
                        #         )
                        # else:
                        #     if key == "L_rec2":
                        #         losses2[key] = np.sum(np.sqrt(losses2[key])) / len(loader.dataset)
                        if key == "L_rec2" or key == "L_efobso":
                            loss_i[r * n_smp_b + i][key] = np.mean(
                                np.sqrt(losses2[key][r * n_smp_b + i])
                            )
                        elif key == "L_angular":
                            # Angular error: mean of degrees (no sqrt needed)
                            loss_i[r * n_smp_b + i][key] = np.mean(
                                losses2[key][r * n_smp_b + i]
                            )
                        elif key in ["L_axis_x", "L_axis_y"]:
                            # Axis errors: RMSE (sqrt of mean squared error)
                            loss_i[r * n_smp_b + i][key] = np.mean(
                                np.sqrt(losses2[key][r * n_smp_b + i])
                            )
                        elif key == "L_angular_rb":
                            # Angular error for rule-based: mean of degrees (no sqrt needed)
                            loss_i[r * n_smp_b + i][key] = np.mean(
                                losses2[key][r * n_smp_b + i]
                            )
                        elif key in ["L_axis_x_rb", "L_axis_y_rb"]:
                            # Axis errors for rule-based: RMSE (sqrt of mean squared error)
                            loss_i[r * n_smp_b + i][key] = np.mean(
                                np.sqrt(losses2[key][r * n_smp_b + i])
                            )

                    # Create the histgram of the player velocities
                    # Save samples
                    experiment_path = "{}/experiments/sample{}".format(
                        experiment_cond, "_eval" if args.bn_eval else ""
                    )
                    os.makedirs(experiment_path, exist_ok=True)
                    fig, ax = plt.subplots()
                    ax.hist(
                        np.sqrt(
                            np.sum(np.square(others["v_gt"][r * n_smp_b + i]), axis=2)
                        ).flatten(),
                        bins=50,
                        range=(0, cfg.PLAYERS_SPEED_LIMIT),
                        alpha=0.5,
                        color="blue",
                        label="Ground truth",
                    )
                    ax.hist(
                        np.sqrt(
                            np.sum(np.square(others["v_pred"][r * n_smp_b + i]), axis=2)
                        ).flatten(),
                        bins=50,
                        range=(0, cfg.PLAYERS_SPEED_LIMIT),
                        alpha=0.5,
                        color="red",
                        label="Prediction",
                    )
                    # ax.set_title("Player velocities: ground truth v.s. prediction", fontsize=16)
                    ax.set_xlabel("Velocity (m/s)", fontsize=16)
                    ax.set_ylabel("Frequency", fontsize=16)
                    ax.set_ylim(0, 12000)
                    plt.xticks(fontsize=16)
                    plt.yticks(fontsize=16)
                    ax.legend(fontsize=16)
                    plt.tight_layout()
                    plt.savefig(
                        "{}/vel_hist_{}.png".format(experiment_path, r * n_smp_b + i)
                    )
                    print("Saved the histogram of the player velocities.")
                    plt.clf()
                    plt.close()

                    # scatter plot of predicted vs true speed
                    if args.model == "GRNN_add":
                        fig, ax = plt.subplots()
                        ax.scatter(
                            np.sqrt(
                                np.sum(
                                    np.square(others["v_gt"][r * n_smp_b + i]), axis=2
                                )
                            ).flatten(),
                            np.sqrt(
                                np.sum(
                                    np.square(others["v_pred"][r * n_smp_b + i]), axis=2
                                )
                            ).flatten(),
                            alpha=0.5,
                            color="black",
                            s=1,
                        )
                        ax.axline(
                            (0, 0),
                            slope=1,
                            color="black",
                            linestyle=(0, (5, 5)),
                        )
                        ax.set_xlabel("True Velocity (m/s)", fontsize=16)
                        ax.set_ylabel("Predicted Velocity (m/s)", fontsize=16)
                        ax.set_xlim(0, cfg.PLAYERS_SPEED_LIMIT)
                        ax.set_ylim(0, cfg.PLAYERS_SPEED_LIMIT)
                        plt.xticks(fontsize=16)
                        plt.yticks(fontsize=16)
                        plt.tight_layout()
                        plt.savefig(
                            "{}/vel_scatter_{}.png".format(
                                experiment_path, r * n_smp_b + i
                            )
                        )
                        print(
                            "Saved the scatter plot of the player velocities (true v.s. predicted)."
                        )
                        plt.clf()
                        plt.close()

                    print(
                        "Test sample "
                        + str(r * n_smp_b + i)
                        + ":\t"
                        + loss_str(loss_i[r * n_smp_b + i], train=-1)
                    )

        epoch_time = time.time() - start_time
        print("Time:\t {:.3f}".format(epoch_time))

        # create Mean + SD for writing a paper------------------------------------------------
        L2_tmp = {}
        for i, d in enumerate(loss_i):
            for key in d:
                if key == "L_rec":
                    continue
                elif key not in L2_tmp:
                    L2_tmp[key] = np.zeros(len(loss_i))
                L2_tmp[key][i] = d[key]

        L2_mean = {}
        L2_std = {}
        L2_best = {}
        for key in L2_tmp:
            L2_mean[key] = np.mean(L2_tmp[key])
            L2_std[key] = np.std(L2_tmp[key])
            if (
                "L_rec2" in key
                or "L_efobso" in key
                or key in ["L_axis_x", "L_axis_y", "L_axis_x_rb", "L_axis_y_rb"]
            ):
                L2_best[key] = np.min(L2_tmp[key])
            elif key in ["L_angular", "L_angular_rb"]:
                L2_best[key] = np.min(L2_tmp[key])
            else:
                continue

        print("model: " + args.model)
        for key in L2_mean:
            if "L_rec2" in key or "L_efobso" in key:
                print(
                    "({})".format(key)
                    + " mean \pm std: "
                    + "{} \pm {}".format(L2_mean[key], L2_std[key])
                    + " best: "
                    + "{}".format(L2_best[key])
                )
            elif key in ["L_angular", "L_angular_rb"]:
                print(
                    "({})".format(key)
                    + " mean \pm std: "
                    + "{:.2f} \pm {:.2f} deg".format(L2_mean[key], L2_std[key])
                    + " best: "
                    + "{:.2f} deg".format(L2_best[key])
                )
            elif key in ["L_axis_x", "L_axis_y", "L_axis_x_rb", "L_axis_y_rb"]:
                print(
                    "({})".format(key)
                    + " mean \pm std: "
                    + "{} \pm {}".format(L2_mean[key], L2_std[key])
                    + " best: "
                    + "{}".format(L2_best[key])
                )

        if not args.model in ["SVM", "RF"]:
            samples_file = experiment_path + "/samples.p"
            pickle.dump(
                [losses, losses2, others, loss_i],
                open(samples_file, "wb"),
                protocol=4,
            )
        else:
            samples_file = experiment_path + "/samples_{}.p".format(args.model)
            pickle.dump(
                [losses2, loss_i],
                open(samples_file, "wb"),
                protocol=4,
            )

        print("Computation is finished!")

        # =====================================================================
        # Test Summary - Saved Files
        # =====================================================================
        print("\n" + "=" * 70)
        print("TEST SUMMARY - SAVED FILES")
        print("=" * 70)
        print(f"Model loaded from : {model_pathname}_best.pth")
        print(f"Results saved to  : {experiment_path}/")
        if args.use_different_test_seeds:
            print(f"Seed mode         : Different seeds (0 to {n_sample - 1})")
        else:
            print(f"Seed mode         : Fixed seed ({args.seed})")
        print("-" * 70)
        print("Saved files:")
        print(f"  - Velocity histograms   : {experiment_path}/vel_hist_*.png")
        if args.model == "GRNN":
            print(f"  - Velocity scatter      : {experiment_path}/vel_scatter_*.png")
        print(f"  - Sample results        : {samples_file}")
        print("-" * 70)
        print("Sample results contain:")
        print("  - v_pred       : Predicted velocities")
        print("  - v_gt         : Ground truth velocities")
        print("  - v_rb         : Rule-based velocities")
        print("  - L_rec2       : Reconstruction loss (RMSE)")
        print("  - L_efobso     : Rule-based baseline loss")
        print("  - L_angular    : Angular error (degrees) - prediction vs ground truth")
        print("  - L_axis_x     : X-axis error - prediction vs ground truth")
        print("  - L_axis_y     : Y-axis error - prediction vs ground truth")
        print("  - L_angular_rb : Angular error (degrees) - rule-based vs ground truth")
        print("  - L_axis_x_rb  : X-axis error - rule-based vs ground truth")
        print("  - L_axis_y_rb  : Y-axis error - rule-based vs ground truth")
        print("=" * 70 + "\n")
