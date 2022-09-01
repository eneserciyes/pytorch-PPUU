import sys
import numpy, random, pdb, math, pickle, glob, time, os, re
import torch

class DataStore:
    def __init__(self, combined_data_path, car_sizes):
        self.random = random.Random()
        self.random.seed(12345)
        self.car_sizes = car_sizes
        self.populated = False
        self.data_path = combined_data_path

        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
        self.ids = []
        self.ego_car_images = []

    def populate(self):
        data = torch.load(self.data_path)
        self.images = data.get("images")
        self.actions = data.get("actions")
        self.costs = data.get("costs")
        self.states = data.get("states")
        self.ids = data.get("ids")
        self.ego_car_images = data.get("ego_car")
        self.populated = True

    def unpopulate(self):
        self.images = []
        self.actions = []
        self.costs = []
        self.states = []
        self.ids = []
        self.ego_car_images = []
        self.populated = False

    def get_batch(self, s, device, T):
        # min is important since sometimes numbers do not align causing issues in stack operation below
        episode_length = min(self.images[s].size(0), self.states[s].size(0))
        if episode_length >= T:
            t = self.random.randint(0, episode_length - T)
            image = self.images[s][t : t + T].to(device)
            action = self.actions[s][t : t + T].to(device)
            state = self.states[s][t : t + T, 0].to(
                device
            )  # discard 6 neighbouring cars
            cost = self.costs[s][t : t + T].to(device)
            iD = self.ids[s]
            ego_car = self.ego_car_images[s].to(device)
            splits = self.ids[s].split("/")
            time_slot = splits[-2]
            car_id = int(re.findall(r"car(\d+).pkl", splits[-1])[0])
            size = self.car_sizes[car_id]
            sizes = [size[0], size[1]]
            return image, state, action, cost, iD, sizes, ego_car

        return None


def save_all_data(data_dir, df, combined_data_path):
    print(data_dir)
    images = []
    actions = []
    costs = []
    states = []
    ids = glob.glob(f"{data_dir}/{df}/car*.pkl")
    ids.sort()
    ego_car_images = []
    for f in ids:
        print(f"[loading {f}]")
        fd = pickle.load(open(f, "rb"))
        Ta = fd["actions"].size(0)
        Tp = fd["pixel_proximity_cost"].size(0)
        Tl = fd["lane_cost"].size(0)
        # assert Ta == Tp == Tl
        # if not(Ta == Tp == Tl): pdb.set_trace()
        images.append(fd["images"])
        actions.append(fd["actions"])
        costs.append(
            torch.cat(
                (
                    fd.get("pixel_proximity_cost")[:Ta].view(-1, 1),
                    fd.get("lane_cost")[:Ta].view(-1, 1),
                ),
                1,
            ),
        )
        states.append(fd["states"])
        ego_car_images.append(fd["ego_car"])

    print(f"Saving {combined_data_path} to disk")
    torch.save(
        {
            "images": images,
            "actions": actions,
            "costs": costs,
            "states": states,
            "ids": ids,
            "ego_car": ego_car_images,
        },
        combined_data_path,
    )
    return len(images)


def setup_data(opt, dataset):
    rdm = random.Random()
    rdm.seed(12345)

    if dataset == "i80" or dataset == "us101":
        data_dir = f"traffic-data/state-action-cost/data_{dataset}_v0"
    else:
        data_dir = dataset
    data_files = next(os.walk(data_dir))[1]

    images = []
    actions = []
    costs = []
    states = []
    ids = []
    ego_car_images = []

    data_store_sizes = [0]
    for df in data_files:
        combined_data_path = f"{data_dir}/{df}/all_data.pth"
        data_store_size = save_all_data(data_dir, df, combined_data_path)
        data_store_sizes.append(data_store_sizes[-1] + data_store_size)
        data = torch.load(combined_data_path)
        images += data.get("images")
        actions += data.get("actions")
        costs += data.get("costs")
        states += data.get("states")
        ids += data.get("ids")
        ego_car_images += data.get("ego_car")
    torch.save(data_store_sizes[1:], data_dir + "/data_store_sizes.pth")
    n_episodes = len(images)
    print("[generating data splits]")
    rgn = numpy.random.RandomState(0)
    perm = rgn.permutation(n_episodes)

    n_train = int(math.floor(n_episodes * 0.8))
    n_valid = int(math.floor(n_episodes * 0.1))

    train_indx = perm[0:n_train]
    valid_indx = perm[n_train : n_train + n_valid]
    test_indx = perm[n_train + n_valid :]

    splits_path = data_dir + "/splits.pth"
    torch.save(
        dict(
            train=train_indx,
            val=valid_indx,
            test=test_indx,
        ),
        splits_path,
    )

    print("[computing action stats]")
    all_actions = []
    for i in train_indx:
        all_actions.append(actions[i])
    all_actions = torch.cat(all_actions, 0)
    a_mean = torch.mean(all_actions, 0)
    a_std = torch.std(all_actions, 0)

    print("[computing state stats]")
    all_states = []
    for i in train_indx:
        all_states.append(states[i][:, 0])
    all_states = torch.cat(all_states, 0)
    s_mean = torch.mean(all_states, 0)
    s_std = torch.std(all_states, 0)

    stats_path = data_dir + "/data_stats_with_diff.pth"
    torch.save(
        {
            "a_mean": a_mean,
            "a_std": a_std,
            "s_mean": s_mean,
            "s_std": s_std,
        },
        stats_path,
    )


def find_data_store_by_index(data_store_sizes, i):
    for j, size in enumerate(data_store_sizes):
        if i < size:
            return j
    raise IndexError


class DataLoader:
    def __init__(self, fname, opt, dataset="simulator", single_shard=False):
        if opt.debug:
            single_shard = True
        self.opt = opt
        self.random = random.Random()
        self.random.seed(
            12345
        )  # use this so that the same batches will always be picked

        if dataset == "i80" or dataset == "us101":
            data_dir = f"traffic-data/state-action-cost/data_{dataset}_v0"
        else:
            data_dir = dataset

        if single_shard:
            # quick load for debugging
            data_files = [f"{next(os.walk(data_dir))[1][0]}"]
        else:
            data_files = next(os.walk(data_dir))[1]

        car_sizes_path = data_dir + "/car_sizes.pth"
        print(f"[loading car sizes: {car_sizes_path}]")
        self.car_sizes = torch.load(car_sizes_path)

        self.data_stores = []
        self.n_episodes = 0
        for data_store_num, df in enumerate(data_files):
            combined_data_path = f"{data_dir}/{df}/all_data.pth"
            self.data_stores.append(
                DataStore(combined_data_path, self.car_sizes[data_files[data_store_num]])
            )

        self.data_store_sizes = torch.load(data_dir + "/data_store_sizes.pth")

        self.n_episodes = self.data_store_sizes[-1]
        print(f"Number of episodes: {self.n_episodes}")

        splits_path = data_dir + "/splits.pth"
        print(f"[loading data splits: {splits_path}]")
        self.splits = torch.load(splits_path)
        self.train_indx = self.splits.get("train")
        self.valid_indx = self.splits.get("val")
        self.test_indx = self.splits.get("test")

        stats_path = data_dir + "/data_stats_with_diff.pth"
        print(f"[loading data stats: {stats_path}]")
        stats = torch.load(stats_path)
        self.a_mean = stats.get("a_mean")
        self.a_std = stats.get("a_std")
        self.s_mean = stats.get("s_mean")
        self.s_std = stats.get("s_std")

    # get batch to use for forward modeling
    # a sequence of ncond given states, a sequence of npred actions,
    # and a sequence of npred states to be predicted
    def get_batch_fm(self, split, npred=-1, cuda=True):

        # Choose the correct device
        device = torch.device("cuda") if cuda else torch.device("cpu")

        if split == "train":
            indx = self.train_indx
        elif split == "valid":
            indx = self.valid_indx
        elif split == "test":
            indx = self.test_indx

        if npred == -1:
            npred = self.opt.npred

        images, states, actions, costs, ids, sizes, ego_cars = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        nb = 0
        T = self.opt.ncond + npred
        while nb < self.opt.batch_size:
            s = self.random.choice(indx)
            ds = find_data_store_by_index(self.data_store_sizes, s)

            if not self.data_stores[ds].populated:
                # check for active stores and disable if needed
                active_data_stores = []
                for i, store in enumerate(self.data_stores):
                    if store.populated:
                        active_data_stores.append(i)
                if len(active_data_stores) >= 2:
                    disable_store_indx = self.random.choice(active_data_stores)
                    self.data_stores[disable_store_indx].unpopulate()
                self.data_stores[ds].populate()

            batch = self.data_stores[ds].get_batch(
                s if ds == 0 else s - self.data_store_sizes[ds - 1], device, T
            )
            if batch:
                image, state, action, cost, iD, size, ego_car = batch
                images.append(image)
                states.append(state)
                actions.append(action)
                costs.append(cost)
                ids.append(iD)
                sizes.append(size)
                ego_cars.append(ego_car)
                nb += 1

        # Pile up stuff
        images = torch.stack(images)
        states = torch.stack(states)
        actions = torch.stack(actions)
        sizes = torch.tensor(sizes, dtype=torch.float32)
        ego_cars = torch.stack(ego_cars)

        # Normalise actions, state_vectors, state_images
        if not self.opt.debug:
            actions = self.normalise_action(actions)
            states = self.normalise_state_vector(states)
        images = self.normalise_state_image(images)
        ego_cars = self.normalise_state_image(ego_cars)

        costs = torch.stack(costs)

        # |-----ncond-----||------------npred------------||
        # ^                ^                              ^
        # 0               t0                             t1
        t0 = self.opt.ncond
        t1 = T
        input_images = images[:, :t0].float().contiguous()
        input_states = states[:, :t0].float().contiguous()
        target_images = images[:, t0:t1].float().contiguous()
        target_states = states[:, t0:t1].float().contiguous()
        target_costs = costs[:, t0:t1].float().contiguous()
        t0 -= 1
        t1 -= 1
        actions = actions[:, t0:t1].float().contiguous()
        # input_actions = actions[:, :t0].float().contiguous()
        ego_cars = ego_cars.float().contiguous()
        #          n_cond                      n_pred
        # <---------------------><---------------------------------->
        # .                     ..                                  .
        # +---------------------+.                                  .  ^          ^
        # |i|i|i|i|i|i|i|i|i|i|i|.  3 × 117 × 24                    .  |          |
        # +---------------------+.                                  .  | inputs   |
        # +---------------------+.                                  .  |          |
        # |s|s|s|s|s|s|s|s|s|s|s|.  4                               .  |          |
        # +---------------------+.                                  .  v          |
        # .                   +-----------------------------------+ .  ^          |
        # .                2  |a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| .  | actions  |
        # .                   +-----------------------------------+ .  v          |
        # .                     +-----------------------------------+  ^          | tensors
        # .       3 × 117 × 24  |i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|i|  |          |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  4  |s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|s|  | targets  |
        # .                     +-----------------------------------+  |          |
        # .                     +-----------------------------------+  |          |
        # .                  2  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|  |          |
        # .                     +-----------------------------------+  v          v
        # +---------------------------------------------------------+             ^
        # |                           car_id                        |             | string
        # +---------------------------------------------------------+             v
        # +---------------------------------------------------------+             ^
        # |                          car_size                       |  2          | tensor
        # +---------------------------------------------------------+             v

        return (
            [input_images, input_states, ego_cars],
            actions,
            [target_images, target_states, target_costs],
            ids,
            sizes,
        )

    @staticmethod
    def normalise_state_image(images):
        return images.float().div_(255.0)

    def normalise_state_vector(self, states):
        shape = (
            (1, 1, 4) if states.dim() == 3 else (1, 4)
        )  # dim = 3: state sequence, dim = 2: single state
        states -= self.s_mean.view(*shape).expand(states.size()).to(states.device)
        states /= (1e-8 + self.s_std.view(*shape).expand(states.size())).to(
            states.device
        )
        return states

    def normalise_action(self, actions):
        actions -= self.a_mean.view(1, 1, 2).expand(actions.size()).to(actions.device)
        actions /= (1e-8 + self.a_std.view(1, 1, 2).expand(actions.size())).to(
            actions.device
        )
        return actions


if __name__ == "__main__":
    # Create some dummy options
    class DataSettings:
        debug = False
        batch_size = 4
        npred = 20
        ncond = 10

    # Instantiate data set object
    d = DataLoader(None, opt=DataSettings, dataset="i80")
    # Retrieve first training batch
    x = d.get_batch_fm("train", cuda=False)
