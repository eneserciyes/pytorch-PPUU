# In[1]:
import pickle
import ipdb
import torch
import time
from dataloader import DataLoader
from team_code.utils import get_opt, set_seed_and_flags


def load_model(mfile):
    print(f"[loading previous checkpoint: {mfile}]")
    torch.nn.Module.dump_patches = True
    model = torch.load(mfile)
    model.cuda()
    model.intype("gpu")
    print("model loaded")
    return model


def predict(dloader, model, npred, z_dropout):
    inputs, actions, targets, _, _ = dloader.get_batch_fm("train", npred)

    pred, loss_p = model(inputs[:-1], actions, targets, z_dropout=z_dropout)
    with open(f"inputs_long_{npred}.pkl", "wb") as f:
        pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"preds_long_{npred}.pkl", "wb") as f:
        pickle.dump(pred, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("prediction done")
    # ipdb.set_trace()
    return pred, loss_p


def main():
    opt = get_opt()
    set_seed_and_flags(opt.seed)

    dataloader = DataLoader(None, opt, opt.dataset, single_shard=True)
    print("data loaded")

    mfile = opt.model_file + ".step200000.model"
    model = load_model(mfile)

    predict(dataloader, model, npred=opt.npred, z_dropout=opt.z_dropout)

    return opt, mfile, dataloader


if __name__ == "__main__":
    print("Starting at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("Ending at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
