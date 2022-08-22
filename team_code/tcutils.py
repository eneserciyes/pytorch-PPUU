import pickle

import argparse
import numpy
import random
import torch

def set_seed_and_flags(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--v", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="i80")
    parser.add_argument("--model", type=str, default="fwd-cnn")
    parser.add_argument(
        "--layers", type=int, default=3, help="layers in frame encoder/decoders"
    )
    parser.add_argument(
        "--data_dir", type=str, default="traffic-data/state-action-cost/data_i80_v0/"
    )
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument(
        "--ncond", type=int, default=20, help="number of conditioning frames"
    )
    parser.add_argument(
        "--npred",
        type=int,
        default=20,
        help="number of predictions to make with unrolled fwd model",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--nfeature", type=int, default=256)
    parser.add_argument(
        "--beta", type=float, default=0.0, help="coefficient for KL term in VAE"
    )
    parser.add_argument("--ploss", type=str, default="hinge")
    parser.add_argument(
        "--z_dropout", type=float, default=0.0, help="set z=0 with this probability"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="regular dropout")
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--lrt", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--epoch_size", type=int, default=2000)
    parser.add_argument(
        "--warmstart", type=int, default=0, help="initialize with pretrained model"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enables tensorboard logging."
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="models",
        help="path to the directory where to save tensorboard log. If passed empty path"
        " no logs are saved.",
    )
    return parser


def get_opt() -> argparse.Namespace:
    parser = get_parser()
    opt = parser.parse_args(
        [
            "--model",
            "fwd-cnn-vae-fp",
            "--layers",
            "3",
            "--batch_size",
            "64",
            "--ncond",
            "20",
            "--npred",
            "20",
            "--lrt",
            "0.0001",
            "--nfeature",
            "256",
            "--dropout",
            "0.1",
            "--nz",
            "32",
            "--beta",
            "1e-06",
            "--z_dropout",
            "0.5",
            "--grad_clip",
            "5",
            "--warmstart",
            "1",
        ]
    )

    opt.model_file = f"{opt.model_dir}/model={opt.model}-layers={opt.layers}-bsize={opt.batch_size}-ncond={opt.ncond}-npred={opt.npred}-lrt={opt.lrt}-nfeature={opt.nfeature}-dropout={opt.dropout}"
    if "vae" in opt.model:
        opt.model_file += f"-nz={opt.nz}"
        opt.model_file += f"-beta={opt.beta}"
        opt.model_file += f"-zdropout={opt.z_dropout}"

    if opt.grad_clip != -1:
        opt.model_file += f"-gclip={opt.grad_clip}"

    opt.model_file += f"-warmstart={opt.warmstart}"
    opt.model_file += f"-seed={opt.seed}"
    print(f"[will save model as: {opt.model_file}]")

    opt.n_inputs = 4
    opt.n_actions = 2
    opt.height = 117
    opt.width = 24
    if opt.layers == 3:
        opt.h_height = 14
        opt.h_width = 3
    elif opt.layers == 4:
        opt.h_height = 7
        opt.h_width = 1
    opt.hidden_size = opt.nfeature * opt.h_height * opt.h_width
    return opt


def load_pickle_data(root="/home/erciyes/Projects/pytorch-PPUU"):
    with open(f"{root}/team_code/tmp/inputs.pkl", "rb") as f:
        inputs = pickle.load(f)
    with open(f"{root}/team_code/tmp/actions.pkl", "rb") as f:
        actions = pickle.load(f)
    with open(f"{root}/team_code/tmp/targets.pkl", "rb") as f:
        targets = pickle.load(f)
    with open(f"{root}/team_code/tmp/ids.pkl", "rb") as f:
        ids = pickle.load(f)
    with open(f"{root}/team_code/tmp/sizes.pkl", "rb") as f:
        sizes = pickle.load(f)
    return inputs, actions, targets, ids, sizes
