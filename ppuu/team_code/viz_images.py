import torch
import time
from plot_lib import set_default
import matplotlib.pyplot as plt

set_default()


def main():
    images = torch.load("tmp/images_input.pth", map_location=torch.device("cpu"))
    viz_rgb = (images[0, 0, :3]).permute(1, 2, 0).detach().numpy()
    plt.imshow(viz_rgb)
    plt.savefig("tmp/viz_rgb.png")


def viz_fourth_channel():
    images = torch.load("tmp/images_input.pth", map_location=torch.device("cpu"))
    viz_fourth = (images[0, 0, 3]).detach().numpy()
    plt.imshow(viz_fourth)
    plt.savefig("tmp/viz_fourth.png")


if __name__ == "__main__":
    print("Starting at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    viz_fourth_channel()
    # Fourth channel is the ego vehicle mask
    print("Ending at", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
