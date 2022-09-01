import utils
import random
import numpy
import torch
from dataloader import DataLoader
import time


def main(opt):
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace(
    #     "localhost", port=5724, stdoutToServer=True, stderrToServer=True
    # )
    start = time.time()
    dataloader = DataLoader(None, opt, opt.dataset)
    for i in range(1):
        print(f"##### BATCH {i} #####")
        inputs, actions, targets, ids, car_sizes = dataloader.get_batch_fm(
            "train", opt.npred
        )
        print(ids)
    print("Took:", time.time() - start)


if __name__ == "__main__":
    opt = utils.parse_command_line()

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    main(opt)