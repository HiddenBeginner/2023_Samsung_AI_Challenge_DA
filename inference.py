import os
from argparse import ArgumentParser

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from segformers.detectors import Backgroud_detector
from segformers.inference import slide_inference
from segformers.networks import SegFormer
from segformers.utils import rle_encode


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dir_data', type=str, default='./data')
    parser.add_argument('--path_ckpt', type=str, default='./ckpt/1695288341/last_ckpt')
    parser.add_argument('--submission_name', type=str, default='./SegFormer-b5')
    args = parser.parse_args()

    return args


def main():
    # Set model
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dict = torch.load(args.path_ckpt)
    model = SegFormer
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)

    df = pd.read_csv(os.path.join(args.dir_data, 'test.csv'))

    result = []
    model.eval()
    for idx in tqdm(range(len(df))):
        img_path = os.path.join(args.dir_data, df.loc[idx, 'img_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        outside = Backgroud_detector(image)[np.newaxis, np.newaxis]
        outside = torch.as_tensor(outside).float().to(device)
        outside = torch.nn.functional.interpolate(
                    outside,
                    size=(540, 960),
                    mode='nearest'
            )
        outside = outside[0][0].cpu().numpy()

        image = cv2.resize(image, (960, 540))
        image = A.Normalize()(image=image)['image']
        with torch.no_grad():
            images = torch.as_tensor(image, dtype=torch.float, device=device).permute(2, 0, 1).unsqueeze(0)
            logits = slide_inference(images, model, stride=(50, 50), crop_size=(512, 512))
            _, predictions = logits.max(1)

        predictions = predictions[0].cpu().numpy()
        predictions[np.where(outside == 1)] = 12
        predictions = predictions.astype(np.int32)
        # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
        for class_id in range(12):
            class_mask = (predictions == class_id).astype(np.int32)
            if np.sum(class_mask) > 0:  # 마스크가 존재하는 경우 encode
                mask_rle = rle_encode(class_mask)
                result.append(mask_rle)
            else:  # 마스크가 존재하지 않는 경우 -1
                result.append(-1)

    submit = pd.read_csv(os.path.join(args.dir_data, 'sample_submission.csv'))
    submit['mask_rle'] = result
    submit.to_csv(args.submission_name, index=False)
