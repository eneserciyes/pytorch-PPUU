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

from ppuu import planning
from ppuu import utils
from ppuu.data.dataloader import DataLoader
from ppuu.models import FwdCNN_VAE, CostPredictor

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#################################################
# Train a policy / controller
#################################################


def start(model, dataloader, optimizer, opt, what, nbatches, npred):
    train = True if what == "train" else False
    model.train()
    model.policy_net.train()
    n_updates, grad_norm = 0, 0
    total_losses = dict(
        proximity=0,
        uncertainty=0,
        lane=0,
        offroad=0,
        action=0,
        policy=0,
        goal=0,
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
            + opt.lambda_g
            * pred["goal"]
            * 2  # goal cost is multiplied by 2 to get approx same scale
        )

        if not math.isnan(pred["policy"].item()):
            if train:
                optimizer.zero_grad()
                pred["policy"].backward()  # back-propagation through time!
                grad_norm += utils.grad_norm(model.policy_net).item()
                torch.nn.utils.clip_grad_norm_(
                    model.policy_net.parameters(), opt.grad_clip
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


def setup_model_and_data(opt):
    # create the model
    model = FwdCNN_VAE(opt)
    model.create_policy_net(opt)
    optimizer = optim.Adam(
        model.policy_net.parameters(), opt.lrt
    )  # POLICY optimiser ONLY!

    model.opt.lambda_l = opt.lambda_l  # used by planning.py/compute_uncertainty_batch
    model.opt.lambda_o = opt.lambda_o  # used by planning.py/compute_uncertainty_batch

    # Load normalisation stats
    stats = torch.load("traffic-data/state-action-cost/data_i80_v0/data_stats.pth")
    model.stats = stats  # used by planning.py/compute_uncertainty_batch

    # Send to GPU if possible
    model.to(opt.device)
    model.policy_net.stats_d = {}
    for k, v in stats.items():
        if isinstance(v, torch.Tensor):
            model.policy_net.stats_d[k] = v.to(opt.device)

    # load the model
    model_path = path.join(opt.model_dir, opt.mfile)
    if path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(state_dict=checkpoint["model"])
        optimizer.load_state_dict(state_dict=checkpoint["opt"])
    else:
        raise RuntimeError(f"couldn't find file {opt.mfile}")

    # if not hasattr(model.encoder, "n_channels"):
    #     model.encoder.n_channels = 3

    if opt.learned_cost:
        print("[loading cost regressor]")
        cost_model = CostPredictor(opt)
        cost_checkpoint = torch.load(
            path.join(opt.model_dir, opt.mfile + ".cost.model")
        )
        cost_model.load_state_dict(cost_checkpoint["model"])
        model.cost = cost_model

    dataloader = DataLoader(None, opt, opt.dataset)
    model.train()
    model.opt.u_hinge = opt.u_hinge
    planning.estimate_uncertainty_stats(
        model, dataloader, n_batches=50, npred=opt.npred
    )
    model.eval()
    return model, optimizer, dataloader


def setup_options():
    opt = utils.parse_command_line()

    # Create file_name
    opt.model_file = path.join(opt.model_dir, "policy_networks", "MPUR-" + opt.policy)
    utils.build_model_file_name(opt)

    # Define default device
    opt.device = torch.device(
        "cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu"
    )
    return opt


def save_checkpoint(name, model, optimizer, opt, n_iter):
    model.to("cpu")
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            opt=opt,
            n_iter=n_iter,
        ),
        name,
    )


def main(model, optimizer, dataloader, opt):
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
    )

    run_name = opt.name if opt.name else None
    wandb.init(
        project="mpur-ppuu",
        name=run_name,
        mode="offline" if opt.name == "debug" else "online",
    )
    wandb.config.update(opt)

    best_loss = float("inf")
    for i in range(250):
        n_iter += opt.epoch_size
        log_string = f"step {n_iter} | "
        train_losses = start(
            model,
            dataloader,
            optimizer,
            opt,
            "train",
            opt.epoch_size if opt.name != "debug" else 1,
            opt.npred,
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
                    model,
                    dataloader,
                    optimizer,
                    opt,
                    "valid",
                    opt.epoch_size // 2 if opt.name != "debug" else 1,
                    opt.npred,
                )
            if valid_losses["policy"] < best_loss:
                best_loss = valid_losses["policy"]
                save_checkpoint(
                    opt.model_file + "best.model", model, optimizer, opt, n_iter
                )

            wandb.log(
                {f"Loss/valid_{key}": value for key, value in valid_losses.items()},
                step=i,
            )
            log_string += (
                "valid: ["
                + ", ".join(f"{k}: {valid_losses[v]:.4f}" for k, v in losses.items())
                + "]"
            )

        print(log_string)
        utils.log(opt.model_file + ".log", log_string)

        save_checkpoint(opt.model_file + ".model", model, optimizer, opt, n_iter)

        if (n_iter / opt.epoch_size) % 20 == 0:
            save_checkpoint(
                opt.model_file + f"step{n_iter}.model", model, optimizer, opt, n_iter
            )

        model.to(opt.device)


if __name__ == "__main__":

    OPT = setup_options()

    os.system("mkdir -p " + path.join(OPT.model_dir, "policy_networks"))
    random.seed(OPT.seed)
    numpy.random.seed(OPT.seed)
    torch.manual_seed(OPT.seed)

    if OPT.pydevd:
        import pydevd_pycharm

        pydevd_pycharm.settrace(
            "localhost", port=5724, stdoutToServer=True, stderrToServer=True
        )

    MODEL, OPTIMIZER, DATALOADER = setup_model_and_data(OPT)
    main(MODEL, OPTIMIZER, DATALOADER, OPT)
