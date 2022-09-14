import math
from collections import OrderedDict
from typing import Tuple

import numpy
import os
import ipdb
import random
import torch
import torch.optim as optim
from os import path
import wandb
import tqdm

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a policy / controller
#################################################

opt = utils.parse_command_line()

if opt.pydevd:
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        "localhost", port=5724, stdoutToServer=True, stderrToServer=True
    )

# Create file_name
opt.model_file = path.join(opt.model_dir, "policy_networks", "MPUR-" + opt.policy)
utils.build_model_file_name(opt)

os.system("mkdir -p " + path.join(opt.model_dir, "policy_networks"))

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Define default device
opt.device = torch.device(
    "cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu"
)
if torch.cuda.is_available() and opt.no_cuda:
    print(
        "WARNING: You have a CUDA device, so you should probably run without -no_cuda"
    )

# load the model

model_path = path.join(opt.model_dir, opt.mfile)
if path.exists(model_path):
    model = torch.load(model_path)
elif path.exists(opt.mfile):
    model = torch.load(opt.mfile)
else:
    raise RuntimeError(f"couldn't find file {opt.mfile}")

if type(model) is dict:
    model = model["model"]

if not hasattr(model.encoder, "n_channels"):
    model.encoder.n_channels = 3

model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch

assert hasattr(model, "goal_policy_net"), "Model must have a goal policy net."
assert hasattr(model, "policy_net"), "Model must have a low level policy net."
# assert hasattr(model, "value_net"), "Model must have a value network."

model.create_value_net(opt)
optimizer = optim.Adam(model.goal_policy_net.parameters(), opt.lrt)

# Load normalisation stats
stats = torch.load("traffic-data/state-action-cost/data_i80_v0/data_stats.pth")
model.stats = stats  # used by planning.py/compute_uncertainty_batch

# Send to GPU if possible
model.to(opt.device)
model.policy_net.stats_d = {}
for k, v in stats.items():
    if isinstance(v, torch.Tensor):
        model.policy_net.stats_d[k] = v.to(opt.device)


dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float("inf")
        self.minimum = max_value_bound if max_value_bound else float("inf")

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, root=False):
        self.root = root
        self.image = None
        self.state = None
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, model, exploration_noise=False):
        # TODO: sample goals
        # TODO: add exploration noise
        # TODO: rollout to create the children nodes
        pass


class MCTS:
    def __init__(self, config, exploration=False):
        self.config = config
        self.exploration = exploration

    def run(self, root, model, num_simulations=50):
        min_max_stats = MinMaxStats()
        for _ in range(num_simulations):
            node = root
            search_path = [node]
            while node.expanded():
                action_cap, node = self.select_child(model, node, min_max_stats)
                search_path.append(node)



    def select_child(self, model, node, min_max_stats):
        # TODO: return goal and nodes
        return None, None


def rollout(
    input_images, input_states, input_ego_car, init_goals, nsteps=5
) -> Tuple[torch.Tensor, torch.Tensor]:
    Z = model.sample_z(nsteps, method="fp")
    Z = Z.view(nsteps, 1, -1)

    assert nsteps > 0, "rollout length must be greater than zero!"
    pred_image, pred_state = None, None
    for t in range(nsteps):
        z_t = Z[t]
        a, entropy, mu, std = model.policy_net(
            input_images,
            input_states,
            goals=init_goals,
            sample=True,
            normalize_goals=False,
        )
        pred_image, pred_state = model.forward_single_step(
            input_images[:, :, :3].contiguous(),
            input_states,
            a,
            z_t,
        )
        pred_image = torch.cat((pred_image, input_ego_car), dim=2)
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)

    return pred_image, pred_state


def preprocess_inputs(inputs):
    input_images_orig, input_states_orig, input_ego_car_orig = inputs
    ego_car_new_shape = [*input_images_orig.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(ego_car_new_shape)

    input_images = torch.cat((input_images_orig, input_ego_car), dim=2)
    return input_images, input_states_orig, input_ego_car_orig[:, :1]


def train(nbatches, npred):
    for j in tqdm.tqdm(range(nbatches)):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(
            "train", npred
        )

        # do preprocessing
        input_images, input_states, input_ego_car = preprocess_inputs(inputs)

        # start search


print("[training]")
