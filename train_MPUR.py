import math
from collections import OrderedDict

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

if opt.goal_rollout_len == -1:
    opt.goal_rollout_len = opt.npred
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
if opt.value_model != "":
    value_function = torch.load(
        path.join(opt.model_dir, "value_functions", opt.value_model)
    ).to(opt.device)
    model.value_function = value_function

if opt.train_policy == "low_level":
    # Create policy
    model.create_policy_net(opt)
    optimizer = optim.Adam(model.policy_net.parameters(), opt.lrt)  # POLICY optimiser ONLY!
elif opt.train_policy == "goal":
    assert model.policy_net, "policy net is not trained"
    model.create_goal_net(opt)
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

if opt.learned_cost and opt.train_policy == "low_level":
    print("[loading cost regressor]")
    model.cost = torch.load(path.join(opt.model_dir, opt.mfile + ".cost.model"))[
        "model"
    ]

dataloader = DataLoader(None, opt, opt.dataset)
model.train()
model.opt.u_hinge = opt.u_hinge
planning.estimate_uncertainty_stats(model, dataloader, n_batches=50, npred=opt.npred)
model.eval()


def start(what, nbatches, npred, epoch):
    train = True if what == "train" else False
    model.train()
    model.policy_net.train()
    if model.goal_policy_net:
        model.goal_policy_net.train()
    n_updates, grad_norm = 0, 0
    total_losses = dict(
        proximity=0,
        uncertainty=0,
        lane=0,
        offroad=0,
        action=0,
        policy=0,
        goal=0,
        goal_predictor=0,
    )
    for j in tqdm.tqdm(range(nbatches)):
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(what, npred)
        pred, actions = planning.train_policy_net_mpur(
            model,
            inputs,
            targets,
            car_sizes,
            goal_distance=opt.goal_distance,
            goal_rollout_len=opt.goal_rollout_len,
            index=(epoch * nbatches) + j,
            n_models=10,
            lrt_z=opt.lrt_z,
            n_updates_z=opt.z_updates,
            infer_z=opt.infer_z,
        )
        pred["policy"] = (
            opt.lambda_p * pred["proximity"]
            + opt.u_reg * pred["uncertainty"]
            + opt.lambda_l * pred["lane"]
            + opt.lambda_a * pred["action"]
            + opt.lambda_o * pred["offroad"]
            # goal cost is multiplied by 2 to get approx same scale
            + opt.lambda_g * pred["goal"] * 2
            + opt.lambda_gp * pred["goal_predictor"]
        )

        updated_model = model.goal_policy_net if opt.train_policy == 'goal' else model.policy_net
        if not math.isnan(pred["policy"].item()):
            if train:
                optimizer.zero_grad()
                pred["policy"].backward()  # back-propagation through time!
                grad_norm += utils.grad_norm(updated_model).item()
                torch.nn.utils.clip_grad_norm_(
                    updated_model.parameters(), opt.grad_clip
                )
                optimizer.step()
            for loss in total_losses:
                total_losses[loss] += pred[loss].item()
            n_updates += 1
        else:
            print("warning, NaN")  # Oh no... Something got quite fucked up!
            ipdb.set_trace()

        if j == 0 and opt.save_movies and train:
            # save videos of normal and adversarial scenarios
            for b in range(opt.batch_size):
                state_img = pred["state_img"][b]
                state_vct = pred["state_vct"][b]
                utils.save_movie(
                    opt.model_file + f".mov/sampled/mov{b}",
                    state_img,
                    state_vct,
                    None,
                    actions[b],
                )

        del inputs, actions, targets, pred

    for loss in total_losses:
        total_losses[loss] /= n_updates
    if train:
        print(f"[avg grad norm: {grad_norm / n_updates:.4f}]")
    return total_losses


print("[training]")
utils.log(opt.model_file + ".log", f"[job name: {opt.model_file}]")
utils.log(opt.model_file + ".log", f"Options used: {opt}")
n_iter = 0
losses = OrderedDict(
    p="proximity",
    l="lane",
    o="offroad",
    u="uncertainty",
    a="action",
    g="goal",
    π="policy",
    gp="goal_predictor",
)

# writer = utils.create_tensorboard_writer(opt)
run_name = opt.name if opt.name else None
wandb.init(project="mpur-ppuu", name=run_name)
wandb.config.update(opt)

best_loss = float("inf")
# set task costs to 0 at first
opt.lambda_l = 0.0
opt.lambda_g = 0.0
opt.lambda_p = 0.0
opt.lambda_gp = 1.0

for i in range(250):
    n_iter += opt.epoch_size
    log_string = f"step {n_iter} | "
    train_losses = start(
        "train", opt.epoch_size if opt.name != "debug" else 1, opt.npred, i
    )
    wandb.log(
        {f"Loss/train_{key}": value for key, value in train_losses.items()}, step=i
    )
    log_string += (
        "train: ["
        + ", ".join(f"{k}: {train_losses[v]:.4f}" for k, v in losses.items())
        + "] | "
    )
    if (i + 1) % 5 == 0:
        with torch.no_grad():  # Torch, please please please, do not track computations :)
            valid_losses = start(
                "valid", opt.epoch_size // 2 if opt.name != "debug" else 1, opt.npred, i
            )
        if valid_losses["policy"] < best_loss:
            best_loss = valid_losses["policy"]
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

        wandb.log(
            {f"Loss/valid_{key}": value for key, value in valid_losses.items()}, step=i
        )
        log_string += (
            "valid: ["
            + ", ".join(f"{k}: {valid_losses[v]:.4f}" for k, v in losses.items())
            + "]"
        )

    print(log_string)
    utils.log(opt.model_file + ".log", log_string)

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
    if (n_iter / opt.epoch_size) % 20 == 0:
        torch.save(
            dict(
                model=model,
                optimizer=optimizer.state_dict(),
                opt=opt,
                n_iter=n_iter,
            ),
            opt.model_file + f"step{n_iter}.model",
        )
    if (n_iter / opt.epoch_size) % 100 == 0 and n_iter != 0:
        eval_submit_script = f"sbatch scripts/submit_eval_mpur.slurm policy={opt.model_file}.model name={opt.name}"
        os.system(f"set -k; {eval_submit_script}")

    model.to(opt.device)

    # set the curriculum
    if i == 100:
        opt.lambda_gp = 0.0
        opt.lambda_l = 0.2
        opt.lambda_g = 1.0
        opt.lambda_p = 1.0

