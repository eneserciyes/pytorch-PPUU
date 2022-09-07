import math
import os
import random
from os import path

import numpy
import torch
import torch.optim as optim
import tqdm
import wandb

import planning
import utils
from dataloader import DataLoader

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#################################################
# Train a goal prediction policy with imitation
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

assert model.policy_net, "policy net is not trained"
model.create_goal_net(opt)
optimizer = optim.Adam(model.goal_policy_net.parameters(), opt.lrt)

# goal stats
goal_stats = torch.load("goal_stats.pth").to(opt.device)

# Send to GPU if possible
model.to(opt.device)

dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge
# planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


def normalize_goal_target(gt_goal):
    return (gt_goal - goal_stats[0]) / goal_stats[1]


def unnormalize_goal(goal_pred):
    return (goal_pred * goal_stats[1]) + goal_stats[0]


def train_goal_bc(what, inputs, targets, goal_distance, index):
    input_images_orig, input_states_orig, input_ego_car_orig = inputs
    target_images, target_states, target_costs = targets
    ego_car_new_shape = [*input_images_orig.shape]
    ego_car_new_shape[2] = 1
    input_ego_car = input_ego_car_orig[:, 2][:, None, None].expand(ego_car_new_shape)

    input_images = torch.cat((input_images_orig, input_ego_car), dim=2)
    input_states = input_states_orig.clone()
    bsize = input_images.size(0)
    npred = target_images.size(1)
    import ipdb

    ipdb.set_trace()
    # current position here
    current_position = input_states[:, -1, :2]
    # get a list of goals
    goal_list = target_states[:, goal_distance::goal_distance, :2]
    gt_goal = planning.get_goal(current_position, goal_list) - current_position
    normalized_goal_target = normalize_goal_target(gt_goal)

    current_goal, _, _, _ = model.goal_policy_net(input_images, input_states)
    goal_predictor_cost = torch.nn.functional.mse_loss(
        current_goal, normalized_goal_target
    )
    if index % 100 == 0:
        # unnormalized is still normalized from actual coordinates
        unnormalized_goal_pred = unnormalize_goal(current_goal)
        planning.visualize_goal_input(
            what, input_images, current_goal, index, s_std=model.stats["s_std"]
        )
    return goal_predictor_cost


def start(what, nbatches, npred, epoch):
    train = True if what == "train" else False
    model.train()
    model.policy_net.train()
    if model.goal_policy_net:
        model.goal_policy_net.train()
    total_loss = 0.0
    n_updates = 0
    for j in tqdm.tqdm(range(nbatches)):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(what, npred)
        goal_predictor_cost = train_goal_bc(
            what,
            inputs,
            targets,
            goal_distance=opt.goal_distance,
            index=(epoch * nbatches) + j,
        )
        if not math.isnan(goal_predictor_cost.item()):
            if train:
                optimizer.zero_grad()
                goal_predictor_cost.backward()
                optimizer.step()
            total_loss += goal_predictor_cost
            n_updates += 1
        else:
            print("warning, NaN")
            import ipdb

            ipdb.set_trace()
        wandb.log(
            {f"Loss/{what}_goal_prediction": goal_predictor_cost},
            step=(epoch * nbatches) + j,
        )

    total_loss /= n_updates
    return total_loss


def main():
    print("[training]")
    utils.log(opt.model_file + ".log", f"[job name: {opt.model_file}]")
    utils.log(opt.model_file + ".log", f"Options used: {opt}")
    n_iter = 0
    run_name = opt.name if opt.name else None
    wandb.init(
        project="mpur-ppuu",
        name=run_name,
        mode="offline" if run_name == "debug" else "online",
    )
    wandb.config.update(opt)

    best_loss = float("inf")

    for i in range(250):
        n_iter += opt.epoch_size
        train_loss = start(
            "train", opt.epoch_size if opt.name != "debug" else 1, opt.npred, i
        )
        with torch.no_grad():
            valid_loss = start(
                "valid", opt.epoch_size // 5 if opt.name != "debug" else 1, opt.npred, i
            )
        if valid_loss < best_loss:
            best_loss = valid_loss
            model.to("cpu")
            torch.save(
                dict(
                    model=model,
                    optimizer=optimizer.state_dict(),
                    opt=opt,
                    n_iter=n_iter,
                ),
                opt.model_file + "best.model",
            )

        model.to("cpu")
        torch.save(
            dict(
                model=model,
                optimizer=optimizer.state_dict(),
                opt=opt,
                n_iter=n_iter,
            ),
            opt.model_file + ".model",
        )

        model.to(opt.device)


if __name__ == "__main__":
    main()
