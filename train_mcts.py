import argparse
import math
from typing import List, Dict, NamedTuple, Tuple
from os import path
import tqdm
from operator import itemgetter

import numpy as np
import os
import random
import torch
import torch.optim as optim
from torch.distributions import Normal

from igraph import Graph, EdgeSeq
import plotly.graph_objects as go

import planning
from dataloader import DataLoader
from models import FwdCNN_VAE


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_command_line():
    parser = argparse.ArgumentParser()
    # model loading options
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-name", type=str, default="debug")
    parser.add_argument("-pydevd", action="store_true")
    parser.add_argument(
        "-mfile",
        type=str,
        help="full model used to train with the MCTS",
        default="best_with_value_net_with_goal_policy_net.model",
    )
    parser.add_argument(
        "-model_dir", type=str, help="path to model directory", default="models/"
    )
    parser.add_argument("-u_hinge", type=float, default=0.5)
    parser.add_argument("-epoch_size", type=int, default=500)
    parser.add_argument("-lrt", type=float, default=0.0001, help="learning rate")

    # Dataset options
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-dataset", type=str, default="i80")
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-ncond", type=int, default=20)
    parser.add_argument("-npred", type=int, default=30)

    # MCTS options
    parser.add_argument("-root_diriclet_alpha", type=float, default=0.25)
    parser.add_argument("-root_exploration_fraction", type=float, default=0.25)
    parser.add_argument("-num_simulations", type=int, default=50)
    parser.add_argument("-pb_c_base", type=float, default=19652)
    parser.add_argument("-pb_c_init", type=float, default=1.25)
    parser.add_argument("-gamma", type=float, default=0.99)

    # Cost options
    parser.add_argument("-lambda_l", type=float, default=0.2)
    parser.add_argument("-lambda_p", type=float, default=1.0)
    parser.add_argument("-lambda_g", type=float, default=1.0)
    parser.add_argument("-lambda_gp", type=float, default=1.0)
    parser.add_argument("-lambda_o", type=float, default=0.0)

    return parser.parse_args()


#################################################
# Train a policy / controller
#################################################
def setup():
    opt = parse_command_line()

    # Create file_name
    opt.model_file = path.join(opt.model_dir, "policy_networks", "MCTS-MPUR-")
    opt.model_file += f"-name={opt.name}"

    os.system("mkdir -p " + path.join(opt.model_dir, "MCTS_networks"))

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
    model.opt.lambda_p = opt.lambda_p  # used by FwdCNN_VAE.reward()
    model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch

    assert hasattr(model, "goal_policy_net"), "Model must have a goal policy net."
    assert hasattr(model, "policy_net"), "Model must have a low level policy net."
    assert hasattr(model, "value_net"), "Model must have a value network."
    assert hasattr(model, "cost"), "Model must have a cost network."

    optimizer = optim.Adam(model.goal_policy_net.parameters(), opt.lrt)

    # Load normalisation stats
    stats = torch.load("traffic-data/state-action-cost/data_i80_v0/data_stats.pth")
    model.stats = stats  # used by planning.py/compute_uncertainty_batch

    goal_stats = torch.load("goal_stats.pth").to(opt.device)
    model.goal_stats = goal_stats

    # Send to GPU if possible
    model.to(opt.device)
    model.policy_net.stats_d = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            model.policy_net.stats_d[k] = v.to(opt.device)

    dataloader = DataLoader(None, opt, opt.dataset)
    model.train()
    model.opt.u_hinge = opt.u_hinge
    planning.estimate_uncertainty_stats(
        model, dataloader, n_batches=50, npred=opt.npred
    )
    model.eval()
    return opt, dataloader, model, optimizer


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


class ActorOutput(NamedTuple):
    mu: torch.Tensor
    std: torch.Tensor


def create_tree(root):
    g = Graph()
    g.add_vertices(1)  # add root

    parent_idx = 0
    total_nodes = 0
    unprocessed_nodes = [
        (parent_idx, goal, child) for goal, child in root.children.items()
    ]
    hover_labels = [
        "Id: {} Value: {} Visit Count: {} prior: {}".format(
            0,
            round(root.value(), 2),
            round(root.visit_count, 2),
            (None if root.prior is None else round(root.prior, 2)),
        )
    ]
    while len(unprocessed_nodes) > 0:
        parent_idx, goal, node = unprocessed_nodes.pop(0)
        child_idx = total_nodes + 1
        g.add_vertices(1)
        hover_labels.append(
            "Id: {} Value: {} reward:{} Visit Count: {} Prior: {} Goal:{}".format(
                total_nodes + 1,
                round(node.value(), 2),
                round(node.reward, 2),
                round(node.visit_count, 2),
                None if node.prior is None else round(node.prior, 2),
                goal.detach().cpu().numpy(),
            )
        )
        g.add_edges([(parent_idx, child_idx)])
        g.add_vertices(len(node.children.keys()))
        unprocessed_nodes += [
            (child_idx, _goal, _child) for _goal, _child in node.children.items()
        ]
        total_nodes += 1

    total_nodes += 1  # incr for root index
    lay = g.layout_auto()
    return g, lay, total_nodes, hover_labels

def get_node_edge_positions(g, lay, total_nodes):
    position = {k: lay[k] for k in range(total_nodes)}
    Y = [lay[k][1] for k in range(total_nodes)]
    M = max(Y)

    es = EdgeSeq(g)  # sequence of edges
    E = [e.tuple for e in g.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]
    return Xn, Yn, Xe, Ye, position, M

class Node:
    def __init__(
        self,
        goal_log_prob: float,
        root=False,
    ):
        self.root: bool = root
        self.state: Tuple = None  # Tuple(image, measurement_state, ego_car)
        self.prior = None
        self.visit_count: int = 0
        self.value_sum: int = 0
        self.reward: int = 0
        self.children: Dict = {}
        self.actor_output: ActorOutput = None
        self.goal_log_prob = goal_log_prob

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(
        self,
        model: FwdCNN_VAE,
        actor_output: ActorOutput,
        state,
        reward,
        proposal_action_sample_n=20,
    ) -> None:
        self.state = state
        self.reward = reward
        self.actor_output = actor_output

        _mu = actor_output.mu.repeat(1, proposal_action_sample_n).reshape(
            proposal_action_sample_n, actor_output.mu.shape[1]
        )
        _std = actor_output.std.repeat(1, proposal_action_sample_n).reshape(
            proposal_action_sample_n, actor_output.std.shape[1]
        )
        _actor_output = ActorOutput(_mu, _std)
        goal_policy_dist = model.goal_policy_net.action_dist(_actor_output)
        _goals = model.goal_policy_net.action_sample(_actor_output)
        # TODO: tune noise_dist
        # noise_dist = Normal(
        #     torch.zeros(_goals.shape).to(_goals.device),
        #     torch.ones(_goals.shape).to(_goals.device) * 0.3,
        # )
        # _goals = _goals + noise_dist.sample()

        child_goal_log_prob = -goal_policy_dist.entropy(_goals)

        for child_goal in _goals:
            self.children[child_goal] = Node(0)
        self.update_prior()

    def update_prior(self):
        exp_sum = sum(
            [
                (math.e**0.25) ** (child.goal_log_prob)
                for goal, child in self.children.items()
            ]
        )
        for child in self.children.values():
            child.prior = (math.e**0.25) ** (child.goal_log_prob) / exp_sum

    def show(self):

        #############
        # Create Tree
        #############
        g, lay, total_nodes, hover_labels = create_tree(self)

        ##########
        # Plot
        ##########
        Xn, Yn, Xe, Ye, position, M = get_node_edge_positions(g, lay, total_nodes)
        v_label = list(map(str, range(total_nodes)))
        labels = v_label

        fig = go.Figure()

        def make_annotations(pos, text, font_size=10, font_color="rgb(250,250,250)"):
            L = len(pos)
            if len(text) != L:
                raise ValueError("The lists pos and text must have the same len")
            annotations = []
            for k in range(L):
                annotations.append(
                    dict(
                        text=labels[
                            k
                        ],  # or replace labels with a different list for the text within the circle
                        x=pos[k][0],
                        y=2 * M - position[k][1],
                        xref="x1",
                        yref="y1",
                        font=dict(color=font_color, size=font_size),
                        showarrow=False,
                    )
                )
            return annotations

        fig.add_trace(
            go.Scatter(
                x=Xe,
                y=Ye,
                mode="lines",
                name="action",
                line=dict(color="rgb(210,210,210)", width=1),
                hoverinfo="none",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=Xn,
                y=Yn,
                mode="markers",
                name="state",
                marker=dict(
                    symbol="circle-dot",
                    size=18,
                    color="#6175c1",  # '#DB4551',
                    line=dict(color="rgb(50,50,50)", width=1),
                ),
                text=hover_labels,
                hoverinfo="text",
                opacity=0.8,
            )
        )
        axis = dict(
            showline=False,  # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=True,
            showticklabels=True,
        )

        fig.update_layout(
            annotations=make_annotations(position, v_label),
            font_size=12,
            showlegend=True,
            xaxis=axis,
            yaxis=axis,
            margin=dict(l=40, r=40, b=85, t=100),
            hovermode="closest",
            plot_bgcolor="rgb(248,248,248)",
        )
        fig.show()

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
                goal, node = self.select_child(node, min_max_stats)
                search_path.append(node)

            parent = search_path[-2]
            image, state, ego_car = parent.state
            # unnormalize goal
            goal = goal * model.goal_stats[1] + model.goal_stats[0]
            next_image, next_state, next_ego_car = rollout(
                model, image, state, ego_car, goal
            )

            _, _, mu, std = model.goal_policy_net(next_image, next_state)
            actor_output = ActorOutput(mu, std)
            # TODO: check if value net is correct like this
            value, _, _, _ = model.value_net(next_image, next_state)
            goal = model.goal_policy_net.action_sample(actor_output)
            reward = model.reward(
                next_image[:, -1, :3].view(-1, 3, 117, 24),
                next_state[:, -1].view(-1, 4),
            )

            node.expand(
                model,
                actor_output,
                state=(next_image, next_state, next_ego_car),
                reward=reward.item(),
            )
            self.backpropagate(search_path, value.item(), min_max_stats)

    def select_child(self, node: Node, min_max_stats) -> Node:
        # TODO: return goal and nodes

        _, goal, child = max(
            [
                (self.ucb_score(node, child, min_max_stats), goal, child)
                for goal, child in node.children.items()
            ],
            key=itemgetter(0),
        )
        return goal, child

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pred_image = torch.cat((pred_image, input_ego_car.unsqueeze(0)), dim=2)
        input_images = torch.cat((input_images[:, 1:], pred_image), 1)
        input_states = torch.cat((input_states[:, 1:], pred_state.unsqueeze(1)), 1)

    return input_images, input_states, pred_image[:, :, 3]


def preprocess_inputs(inputs):
    input_images_orig, input_states_orig, input_ego_car_orig = inputs
    ego_car_new_shape = [*input_images_orig.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(ego_car_new_shape)

    input_images = torch.cat((input_images_orig, input_ego_car), dim=2)
    return input_images, input_states_orig, input_ego_car_orig[:, :1]


def train(dataloader, model: FwdCNN_VAE, optimizer, config, epoch_size, npred):
    for j in tqdm.tqdm(range(epoch_size)):
        inputs, actions, targets, _, _ = dataloader.get_batch_fm("train", npred)

        # do preprocessing
        input_image, input_state, input_ego_car = preprocess_inputs(inputs)

        # start search
        _, _, mu, std = model.goal_policy_net(input_image, input_state)
        # unnormalize mu and std
        actor_output = ActorOutput(mu, std)
        child_goal = model.goal_policy_net.action_sample(
            actor_output, deterministic=False
        )
        reward = model.reward(
            input_image[:, -1, :3].view(-1, 3, 117, 24), input_state[:, -1].view(-1, 4)
        )
        actor_dist = model.goal_policy_net.action_dist(actor_output)
        child_goal_log_prob = -actor_dist.entropy(child_goal)

        root = Node(0, root=True)
        root.expand(
            model,
            actor_output=actor_output,
            state=(input_image, input_state, input_ego_car),
            reward=reward.item(),
        )

        # TODO: add exploration noise

        MCTS(config).run(root, model, num_simulations=config.num_simulations)
        root.show()

        child_values = [
            (root.reward + config.gamma * child.value(), goal, child)
            for goal, child in root.children.items()
        ]
        _, greedy_goal, greedy_child = max(child_values)
        visit_counts = [
            (child.visit_count, goal, child) for goal, child in root.children.items()
        ]
        # TODO: add temperature
        temperature = 1
        goal_probs = [
            visit_count_i ** (1 / temperature) for visit_count_i, _, _ in visit_counts
        ]
        total_count = sum(goal_probs)
        goal_probs = [x / total_count for x in goal_probs]

        # TODO: fit goal policy to the new goal probabilities


if __name__ == "__main__":
    OPT, DATALOADER, MODEL, OPTIMIZER = setup()
    print("[training]")

    train(DATALOADER, MODEL, OPTIMIZER, OPT, OPT.epoch_size, OPT.npred)
