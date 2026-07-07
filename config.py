"""Configuration, argument parsing and shared training utilities.

Reorganized from the original ``utils.py`` / ``graphmae/utils.py``.
Only the pieces actually used by the SynCC pipeline are kept; every
numeric helper is preserved verbatim so results are unchanged.
"""
import os
import argparse
import random
import logging
from functools import partial

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def build_args():
    parser = argparse.ArgumentParser(description="SynCC")

    # Run control
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="Quake_Muscle")
    parser.add_argument("--task", type=str, default="cls")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--use_cfg", action="store_true", default=False)
    parser.add_argument("--logging", action="store_true")

    # Checkpoint I/O
    #   --load_model : skip training and evaluate weights from --model_path
    #   --save_model : after training, save weights to --model_path
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--model_path", type=str,
                        default="finalmodel_editDM/last/Quake_Muscle_up.pt",
                        help="path to load/save the model state_dict")

    # Training
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=6607)

    # Model architecture
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_out_heads", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_hidden", type=int, default=256)
    parser.add_argument("--num_projector_hidden", type=int, default=256)
    parser.add_argument("--num_projector", type=int, default=256)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument("--in_drop", type=float, default=.2)
    parser.add_argument("--attn_drop", type=float, default=.1)
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--aggr", type=str, default="node")

    # Masking / augmentation
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--augmentation", type=str, default='degree')
    parser.add_argument("--drop_edge_rate", type=float, default=0.4)
    parser.add_argument("--drop_node_rate", type=float, default=0.4)
    parser.add_argument("--drop_feature_rate", type=float, default=0.1)

    # Loss
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--rec_weight", type=float, default=1.0)
    parser.add_argument("--con_weight", type=float, default=0.2)
    parser.add_argument("--zinb_weight", type=float, default=0.8)

    # Dataset (scRNA-seq specific)
    parser.add_argument('--name', type=str, default='baron_mouse',
                        help='baron_mouse, mouse_es, mouse_bladder, zeisel, baron_human')
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)

    # kNN graph
    parser.add_argument('--k', type=int, default=15)

    # Preprocessing
    parser.add_argument('--HVG', type=int, default=2000)
    parser.add_argument('--sf', action='store_true', default=True)
    parser.add_argument('--log', action='store_true', default=True)
    parser.add_argument('--normal', action='store_true', default=False)

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "rrelu":
        return nn.RReLU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
