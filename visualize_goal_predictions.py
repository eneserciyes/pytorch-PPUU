import random
from os import path

import numpy
import torch
import wandb

from dataloader import DataLoader
from eval_policy import parse_args, load_models
from planning import get_goal, visualize_goal_input


def visualize_goal_predictions(model, batch, goal_stats, index, goal_distance=5):
    inputs, actions, targets, _, _ = batch
    input_images_orig, input_states_orig, input_ego_car_orig = inputs
    target_images, target_states, target_costs = targets
    ego_car_new_shape = [*input_images_orig.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(ego_car_new_shape)

    input_images = torch.cat((input_images_orig, input_ego_car), dim=2)
    input_states = input_states_orig.clone()

    goal_list = target_states[:, goal_distance::goal_distance, :2]
    # current position here
    current_position = input_states[:, -1, :2]
    gt_goal = get_goal(current_position, goal_list) - current_position
    current_goal, _, _, _ = model.goal_policy_net(input_images, input_states)
    if goal_stats is not None:
        # unnormalize goal predictions
        current_goal = ((current_goal * goal_stats[1]) + goal_stats[0]).squeeze()
    import ipdb

    ipdb.set_trace()
    visualize_goal_input(
        "eval", input_images, current_goal, gt_goal.squeeze(), index, s_std=model.stats["s_std"]
    )


def main():
    opt = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    data_path = "traffic-data/state-action-cost/data_i80_v0"
    goal_stats = torch.load("goal_stats.pth").to(device)

    dataloader = DataLoader(None, opt, "i80")
    (
        forward_model,
        value_function,
        policy_network_il,
        policy_network_mper,
        data_stats,
    ) = load_models(opt, data_path, device)

    assert hasattr(
        forward_model, "goal_policy_net"
    ), "model doesn't have a goal predictor"
    splits = torch.load(path.join(data_path, "splits.pth"))

    run_name = "eval_" + opt.name if opt.name else None
    wandb.init(project="goal_predictor_mpur", name=run_name)
    wandb.config.update(opt)

    for i in range(100):
        batch = dataloader.get_batch_fm("train", npred=opt.npred)
        visualize_goal_predictions(
            model=forward_model, batch=batch, goal_stats=goal_stats, index=i
        )


if __name__ == "__main__":
    main()
