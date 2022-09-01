from ppuu.data.dataloader import DataLoader
from ppuu.team_code.tcutils import load_pickle_data, get_opt
import time
import torch
from ppuu.utils import proximity_cost, lane_cost


def main():
    opt = get_opt()
    dataloader = DataLoader(None, opt, opt.dataset, single_shard=True)
    inputs, actions, targets, ids, sizes = load_pickle_data()
    input_images, input_states, ego_cars = inputs
    print("Input images:", input_images.shape)
    print("Input states:", input_states.shape)
    print("Ego cars:", ego_cars.shape)
    print("Car count:", len(ids))
    print("Car sizes:", sizes.shape)
    car_size = torch.tensor([6.4, 14.3])
    ###
    bsize = 1
    proximity_costs, proximity_mask = proximity_cost(
        input_images[0:bsize],
        input_states[0:bsize],
        car_size=car_size.expand(bsize, 2),
        unnormalize=True,
        s_mean=dataloader.s_mean,
        s_std=dataloader.s_std,
    )
    lane_costs, lane_mask = lane_cost(
        input_images[0:bsize],
        car_size=car_size.expand(bsize, 2),
    )
    with open("team_code/tmp/lane_costs.pth", "wb") as f:
        torch.save(lane_costs, f)
    with open("team_code/tmp/lane_mask.pth", "wb") as f:
        torch.save(lane_mask, f)


if __name__ == "__main__":
    print("Starting at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("Ending at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
