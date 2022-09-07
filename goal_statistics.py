import random
from os import path

import numpy
import torch
import tqdm

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

opt = utils.parse_command_line()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

# Create file_name
opt.model_file = path.join(opt.model_dir, "policy_networks", "MPUR-" + opt.policy)
utils.build_model_file_name(opt)

# Define default device
opt.device = "cpu"
dataloader = DataLoader(None, opt, opt.dataset)


def calculate_goal_stats():
    goals = []
    for _ in tqdm.tqdm(range(50000)):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(
            "train", opt.npred
        )
        _, input_states, _ = inputs
        _, target_states, _ = targets
        current_position = input_states[:, -1, :2]
        goal_list = target_states[:, opt.goal_distance :: opt.goal_distance, :2]
        gt_goal = planning.get_goal(current_position, goal_list) - current_position
        goals.append(gt_goal)
    goals = torch.cat(goals, dim=0)
    goal_mean = torch.mean(goals, dim=0)
    goal_std = torch.std(goals, dim=0)
    torch.save(torch.stack([goal_mean, goal_std]).cpu(), "goal_stats.pth")


if __name__ == "__main__":
    calculate_goal_stats()
