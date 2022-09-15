import argparse
import math
from typing import List, Tuple

import numpy as np
import os
import random
import torch
import torch.optim as optim
from os import path
import tqdm

import planning
from dataloader import DataLoader
from models import FwdCNN_VAE

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_command_line():
    parser = argparse.ArgumentParser()
    # model loading options
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-name", type=str)
    parser.add_argument("-dataset", type=str, default="i80")
    parser.add_argument("-pydevd", action="store_true")
    parser.add_argument(
        "-mfile",
        type=str,
        help="full model used to train with the MCTS",
    )
    parser.add_argument("-u_hinge", type=float, default=0.5)
    parser.add_argument("-epoch_size", type=int, default=500)

    # MCTS options
    parser.add_argument("-root_diriclet_alpha", type=float, default=0.25)
    parser.add_argument("-root_exploration_fraction", type=float, default=0.25)
    parser.add_argument("-num_simulations", type=int, default=50)
    parser.add_argument("-pb_c_base", type=float, default=19652)
    parser.add_argument("-pb_c_init", type=float, default=1.25)
    parser.add_argument("-gamma", type=float, default=0.99)

    return parser.parse_args()


#################################################
# Train a policy / controller
#################################################

opt = parse_command_line()

if opt.pydevd:
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        "localhost", port=5724, stdoutToServer=True, stderrToServer=True
    )

# Create file_name
opt.model_file = path.join(opt.model_dir, "policy_networks", "MCTS-MPUR-")
opt.model_file += f"-name={opt.name}"

os.system("mkdir -p " + path.join(opt.model_dir, "policy_networks"))

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Define default device
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model

model_path = path.join(opt.model_dir, opt.mfile)
if path.exists(model_path):
    model = torch.load(model_path)
else:
    raise RuntimeError(f"couldn't find file {opt.mfile}")

if type(model) is dict:
    model = model["model"]

model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch

assert hasattr(model, "goal_policy_net"), "Model must have a goal policy net."
assert hasattr(model, "policy_net"), "Model must have a low level policy net."
assert hasattr(model, "value_net"), "Model must have a value network."
assert hasattr(model, "cost"), "Model must have a cost network."

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
        self.ego_car = None
        self.visit_count = 0
        self.value_sum = 0
        self.children_sample_count = 20
        self.goal_distance = 5
        self.reward = 0
        self.children = {}

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, model: FwdCNN_VAE) -> None:
        goals = model.goal_policy_net(
            self.image, self.state, n_samples=self.children_sample_count
        )  # self.children_sample_count x goal_dimension

        for goal in goals:
            next_image, next_state, next_ego_car = rollout(
                model=model,
                input_images=self.image,
                input_states=self.state,
                input_ego_car=self.ego_car,
                init_goals=goal,
                nsteps=self.goal_distance,
            )
            child_node = Node()
            child_node.image, child_node.state, child_node.ego_car = (
                next_image,
                next_state,
                next_ego_car,
            )
            reward = model.cost(next_image, next_state)
            model.reward = 1 / (1e6 + reward)
            self.children[goal] = child_node

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].state = self.children[a].state * (1 - frac) + n * frac


class MCTS:
    def __init__(self, config):
        self.config = config

    def run(self, root, model, num_simulations=50):
        min_max_stats = MinMaxStats()
        for _ in range(num_simulations):
            node: Node = root
            search_path = [node]
            while node.expanded():
                node = self.select_child(node, min_max_stats)
                search_path.append(node)
            node.expand(model)
            value = model.value_net(node.image, node.state)
            self.backpropagate(search_path, value.item(), min_max_stats)

    def select_child(self, node, min_max_stats) -> Node:
        # TODO: return goal and nodes

        _, child = max(
            (self.ucb_score(node, child, min_max_stats), child)
            for _, child in node.children().items()
        )
        return child

    def ucb_score(self, parent, child, min_max_stats) -> float:
        pb_c = math.log(
            (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
        )
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(
        self, search_path: List[Node], value: float, min_max_stats: MinMaxStats
    ):
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.gamma * value


def rollout(
    model, input_images, input_states, input_ego_car, init_goals, nsteps=5
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: return ego car separately
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


def train(config, epoch_size, npred):
    for j in tqdm.tqdm(range(epoch_size)):
        import ipdb

        ipdb.set_trace()
        inputs, actions, targets, _, _ = dataloader.get_batch_fm(
            "train", npred
        )

        # do preprocessing
        input_images, input_states, input_ego_car = preprocess_inputs(inputs)

        # start search
        root = Node(root=True)
        root.image = input_images,
        root.input_states = input_states,
        root.ego_car = input_ego_car
        
        root.expand()
        root.add_exploration_noise(
            config.root_diriclet_alpha, config.root_exploration_fraction
        )
        MCTS(config).run(root, model, num_simulations=config.num_simulations)


print("[training]")
train(opt, opt.epoch_size, opt.npred)
