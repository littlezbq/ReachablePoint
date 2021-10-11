import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import visdom


def compute_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (C x H/r x W/r)
        targets (C x H/r x W/r)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    # beta=4
    neg_weights = torch.pow(1 - targets, 4).float()

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # 正样本

        # alpha=2
        # print(type(neg_weights))

        neg_loss = torch.log(1 - pred) * torch.pow(pred,2) * neg_weights * neg_inds # 负样本

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)