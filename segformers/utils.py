import datetime
import math
import random

import platform
import pynvml
import psutil
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


def print_env():
    """
    현재 사용중인 실험환경에 대한 기본정보를 프린트합니다.
    다음의 정보를 프린트합니다.
    - 실험날짜 
    - Python version 
    - Pytorch version
    - OS
    - CPU spec
    - RAM spec
    - GPU spec
    """
    print('========== System Information ==========')
    # 오늘의 날짜
    current_date = datetime.date.today()
    print(f'DATE : {current_date}')
    
    # Python 버전
    python_version = platform.python_version()
    print(f'Pyton Version : {python_version}')
    
    # Pytorch 버전
    # 이 환경에 PyTorch가 설치되어 있는지 확인합니다.
    try:
        import torch
        pytorch_version = torch.__version__
        
    except ImportError:
        pytorch_version = "PyTorch not installed"

    print(f'PyTorch Version : {pytorch_version}')
    # 현재 작업환경의 os
    os_info = platform.system() + " " + platform.release()
    print(f'OS : {os_info}')
        
    # 현재 작업환경의 CPU 스펙
    cpu_info = platform.processor()
    print(f'CPU spec : {cpu_info}')
    
    # 현재 작업환경의 Memory 스펙
    mem_info = psutil.virtual_memory().total
    print(f'RAM spec : {mem_info / (1024**3):.2f} GB')
    
    # 현재 작업환경의 GPU 스펙
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        
        print(f"Device {i}:")
        print(f"Name: {name}")
        print(f"Total Memory: {memory_info.total / 1024**2} MB")
        print(f"Driver Version: {driver_version}")
        print("="*30)

    pynvml.nvmlShutdown()