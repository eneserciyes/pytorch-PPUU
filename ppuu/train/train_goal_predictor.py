import argparse
import os
from os import path
import random
import numpy
import torch
from torch import optim

from ppuu import planning
from ppuu import utils
from ppuu.data.dataloader import DataLoader
from ppuu.models import FwdCNN_VAE, CostPredictor
from ppuu.train.train_MPUR import setup_options, main


def setup_model_and_data(opt: argparse.Namespace):
    model_path = path.join(opt.model_dir, opt.mfile)
    checkpoint = torch.load(model_path)

    # create the model
    model = FwdCNN_VAE(checkpoint["opt"])
    model.create_policy_net(checkpoint["opt"])
    if checkpoint["opt"].learned_cost:
        print("[loading cost regressor]")
        cost_model = CostPredictor(checkpoint["opt"])
        model.cost = cost_model

    if "goal_policy_net.encoder.f_encoder.0.weight" in checkpoint["model"].keys():
        model.create_goal_net(opt)
        # only update goal policy network parameters
        optimizer = optim.Adam(model.goal_policy_net.parameters(), opt.lrt)
        model.load_state_dict(checkpoint["model"])
        if checkpoint["optimizer"]:
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        model.load_state_dict(checkpoint["model"])
        model.create_goal_net(opt)
        optimizer = optim.Adam(model.goal_policy_net.parameters(), opt.lrt)

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

    dataloader = DataLoader(None, opt, opt.dataset)
    model.train()
    model.opt.u_hinge = opt.u_hinge
    planning.estimate_uncertainty_stats(
        model, dataloader, n_batches=50, npred=opt.npred
    )
    model.eval()
    return model, optimizer, dataloader


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
