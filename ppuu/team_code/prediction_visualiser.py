import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import experiment_debugger as ed
import cv2
from pathlib import Path
import pickle


# In[48]:


class Renderer(ed.Renderer):
    def render(self, img, pred=False):
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        if img.shape[0] % 2 != 0:
            pad = np.zeros((1,) + img.shape[1:], dtype=np.uint8)
            img = np.concatenate((img, pad), axis=0)

        if pred:
            border_width = 1
            border_color = [255, 0, 0]
            img = cv2.copyMakeBorder(
                img,
                border_width,
                border_width,
                border_width,
                border_width,
                cv2.BORDER_CONSTANT,
                value=border_color,
            )
        return img


renderer = Renderer()
debugger = ed.ExperimentDebugger(
    renderer, make_video=True, video_name="pred_results.mp4"
)


# In[49]:


def main(i):
    with open("inputs_long_100.pkl", "rb") as f:
        inputs = pickle.load(f)[i].cpu().detach().numpy()
    with open("preds_long_100.pkl", "rb") as f:
        pred = pickle.load(f)[i].cpu().detach().numpy()

    #     inputs = torch.load(f"inputs_{i}.pth", map_location=torch.device('cpu')).detach().numpy()
    #     pred = torch.load(f"pred_{i}.pth", map_location=torch.device('cpu')).detach().numpy()
    print("Input:", inputs.shape)
    print("Pred:", pred.shape)
    with debugger:
        for j in range(inputs.shape[1]):
            debugger.debug(inputs[0, j])
        for j in range(pred.shape[1]):
            debugger.debug(pred[0, j], pred=True)


main(i=0)