import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data import datasets
from modeling import KeyPointModel
import matplotlib.pyplot as plt
from dataloader import make_dataLoader


# parser = argparse.ArgumentParser(description="model path")
# parser.add_argument('--model', type=str, default="")
#
# args = parser.parse_args()


def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=1):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def flip_tensor(x):
    return torch.flip(x, [3])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def predict(test_loader, net, saveRoot):
    # img = Image.open(img_path)
    # name = os.path.basename(img_path)
    # label_root = r'F:\Projects\Pytorch\Gxy-Template\data\datasets\Qingmuchuan\labels'
    i = 0
    for batch_data in test_loader:
        img, label = batch_data
        # trans_img = np.float32(np.array(img) / 8000)
        label = np.array(label)

        # trans_img = trans_img.unsqueeze(dim=0)

        net.eval()
        # out = net(trans_img).detach().cpu().numpy()
        out = net(img).detach().numpy()

        out = out.squeeze()
        label = label.squeeze()

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(label, cmap='gist_earth')
        ax[1].imshow(out, cmap='gist_earth')

        plt.savefig(saveRoot + "/output_" + str(i))
        plt.close(fig)
        i += 1


if __name__ == "__main__":
    dataRoot = r"F:\Projects\Pytorch\Gxy-Template\data\datasets\Qingmuchuan"
    saveRoot = r"F:\Projects\Pytorch\Gxy-Template\test_output"
    loadModel = r"F:\Projects\Pytorch\Gxy-Template\runs\logs\epoch_495_0.436.pt"
    model = KeyPointModel()

    model.load_state_dict(torch.load(loadModel))

    _, _, test_loader = make_dataLoader(dataRoot)
    predict(test_loader, model,saveRoot)
