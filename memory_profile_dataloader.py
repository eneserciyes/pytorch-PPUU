import utils
import random
import numpy
import torch
from dataloader import DataLoader

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=5724, stdoutToServer=True, stderrToServer=True)

opt = utils.parse_command_line()

random.seed(opt.seed)
numpy.random.seed(opt.seed)
torch.manual_seed(opt.seed)

opt.device = torch.device(
    "cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu"
)

dataloader = DataLoader(None, opt, opt.dataset)

