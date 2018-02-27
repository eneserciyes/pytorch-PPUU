import torch, numpy, argparse, pdb
import models
from dataloader import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser()
# data params
parser.add_argument('-data_dir', type=str, default='/misc/vlgscratch4/LecunGroup/nvidia-collab/data/')
parser.add_argument('-n_episodes', type=int, default=100)
parser.add_argument('-lanes', type=int, default=3)
parser.add_argument('-ncond', type=int, default=4)
parser.add_argument('-npred', type=int, default=4)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-n_hidden', type=int, default=100)
parser.add_argument('-lrt', type=float, default=0.001)
parser.add_argument('-epoch_size', type=int, default=1000)
opt = parser.parse_args()

data_file = f'{opt.data_dir}/traffic_data_lanes={opt.lanes}-episodes={opt.n_episodes}-seed=*.pkl'
dataloader = DataLoader(data_file, opt)

opt.n_inputs = 4
opt.n_actions = 3

policy = models.PolicyMLP(opt).cuda()
optimizer = optim.Adam(policy.parameters(), opt.lrt)

def train(nbatches):
    policy.train()
    total_loss = 0
    for i in range(nbatches):
        optimizer.zero_grad()
        states, masks, actions = dataloader.get_batch_il('train')
        states = Variable(states)
        masks = Variable(masks)
        actions = Variable(actions)
        pred_a = policy(states, masks)
        loss = F.mse_loss(pred_a, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    return total_loss / nbatches


def test(nbatches):
    policy.eval()
    total_loss = 0
    for i in range(nbatches):
        states, masks, actions = dataloader.get_batch_il('valid')
        states = Variable(states)
        masks = Variable(masks)
        actions = Variable(actions)
        pred_a = policy(states, masks)
        loss = F.mse_loss(pred_a, actions)
        total_loss += loss.data[0]
    return total_loss / nbatches


for _ in range(100):
    train_loss = train(opt.epoch_size)
    valid_loss = test(opt.epoch_size)
    print(f'train loss: {train_loss}, test loss: {valid_loss}')
