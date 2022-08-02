# In[1]:
import pickle

import ipdb
import torch, numpy, argparse, pdb, os, time, math, random

# import experiment_debugger

# import utils
from dataloader import DataLoader

print("Starting at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# In[2]:


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


def get_opt():
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


# In[6]:


def main(opt):
    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    dataloader = DataLoader(None, opt, opt.dataset, single_shard=True)

    mfile = opt.model_file + ".step200000.model"

    return opt, mfile, dataloader


OPT, mfile, dataloader = main(get_opt())
print("data loaded")


# In[7]:


def load_model(mfile):
    print(f"[loading previous checkpoint: {mfile}]")
    torch.nn.Module.dump_patches = True
    model = torch.load(mfile)
    model.cuda()
    model.intype("gpu")
    return model


model = load_model(OPT, mfile)
print("model loaded")


# In[10]:


def predict(dloader, model, npred=OPT.npred):
    inputs, actions, targets, _, _ = dloader.get_batch_fm("train", npred)
    # with open("inputs.pkl", "rb") as f:
    #     inputs = pickle.load(f)
    # with open("actions.pkl", "rb") as f:
    #     actions = pickle.load(f)
    # with open("targets.pkl", "rb") as f:
    #     targets = pickle.load(f)
    pred, loss_p = model(inputs[:-1], actions, targets, z_dropout=OPT.z_dropout)
    with open(f"inputs_long_{npred}.pkl", "wb") as f:
        pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"preds_long_{npred}.pkl", "wb") as f:
        pickle.dump(pred, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("prediction done")
    # ipdb.set_trace()


predict(dataloader, model, npred=100)

# In[ ]:
