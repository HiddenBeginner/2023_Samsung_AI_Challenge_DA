import math
import random

import numpy as np
import torch
from matplotlib.colors import ListedColormap

label_dict = {
    0: 'Road',
    1: 'Sidewalk',
    2: 'Construction',
    3: 'Fence',
    4: 'Pole',
    5: 'Traffic Light',
    6: 'Traffic Sign',
    7: 'Nature',
    8: 'Sky',
    9: 'Person',
    10: 'Rider',
    11: 'Car',
    12: 'Unknown'
}

hex_lst = ['#FF0000', '#00FFFF', '#214525', '#EF2FAC', '#D4AC20', '#668DD5',
           '#FF5F15', '#0932DD',  '#800000', '#7DC79F', '#683987', '#000000', '#00FF00']

custom_cmap = ListedColormap(hex_lst)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def compute_iou(pred, target, num_classes):
    iou_list = []
    pred = pred.view(-1)
    target = target.view(-1)

    # For classes excluding the background
    for cls in range(num_classes - 1):  # We subtract 1 to exclude the background class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum().float()
        union = (pred_inds + target_inds).sum().float()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            iou_list.append((intersection / union).item())
    return iou_list


def compute_mIoU(preds, labels, num_classes=13):
    iou_list = compute_iou(preds, labels, num_classes)
    valid_iou_list = [iou for iou in iou_list if not math.isnan(iou)]
    mIoU = sum(valid_iou_list) / len(valid_iou_list)
    return mIoU
