import utils
import random
import numpy
import torch
from dataloader import DataLoader
import time
from memory_profiler import profile

@profile
def main(opt):
    # import pydevd_pycharm
    #
    # pydevd_pycharm.settrace(
    #     "localhost", port=5724, stdoutToServer=True, stderrToServer=True
    # )
    start = time.time()
    dataloader = DataLoader(None, opt, opt.dataset)
    for i in range(5):
        print(f"##### BATCH {i} #####")
        batch = dataloader.get_batch_fm("train", opt.npred)
        inputs, actions, targets, ids, car_sizes = batch
        # print("Inputs 0 hash", torch.mean(inputs[0]))
        # print("Inputs 1 hash", torch.mean(inputs[1]))
        # print("Inputs 2 hash", torch.mean(inputs[2]))
        # print("Actions hash", torch.mean(actions))
        # print("Targets 0 hash", torch.mean(targets[0]))
        # print("Targets 1 hash", torch.mean(targets[1]))
        # print("Targets 2 hash", torch.mean(targets[2]))
    print("Took:", time.time() - start)


if __name__ == "__main__":
    opt = utils.parse_command_line()

    random.seed(opt.seed)
    numpy.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    main(opt)

