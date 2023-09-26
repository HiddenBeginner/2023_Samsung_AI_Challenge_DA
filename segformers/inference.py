import numpy as np
import torch
import torch.nn.functional as F

from .detectors import Backgroud_detector


class Inference:
    def __init__(
        self,
        model,
        image_processor,
        crop_scheme=False
    ):
        self.model = model
        self.image_processor = image_processor
        self.crop_scheme = crop_scheme
        self.model.eval()

    def __call__(self, x):
        return self.predict(x)

    def predict(self, image):
        if self.crop_scheme:
            return self._crop_predict(image)
        else:
            return self._predict(image)

    @torch.no_grad()
    def _predict(self, image):
        inputs = self.image_processor(image, return_tensors='pt')

        x = inputs['pixel_values'].float()
        logits = self.model(pixel_values=x)[0]  # 128 x 128
        logits = torch.nn.functional.interpolate(logits, size=(540, 960), mode="bilinear", align_corners=False)
        mask = torch.argmax(logits, dim=1)
        mask = mask[0].cpu()
        mask = self.remove_outside(image, mask)

        return mask

    @torch.no_grad()
    def _crop_predict(self, image):
        left = image[:, :960]
        right = image[:, 960:]
        inputs = np.stack((left, right))
        inputs = self.image_processor(inputs, return_tensors='pt')

        x = inputs['pixel_values'].float()
        logits = self.model(pixel_values=x)[0]  # 128 x 128
        left, right = logits[0], logits[1]
        logits = torch.zeros((1, 13, 128, 256))
        logits[:, :, :, :128] = left
        logits[:, :, :, 128:] = right
        logits = torch.nn.functional.interpolate(logits, size=(540, 960), mode="bilinear", align_corners=False)
        mask = torch.argmax(logits, dim=1)
        mask = mask[0].cpu()
        mask = self.remove_outside(image, mask)

        return mask

    def remove_outside(self, image, mask):
        outside = Backgroud_detector(image)
        outside = torch.as_tensor(outside[np.newaxis, np.newaxis])
        outside = torch.nn.functional.interpolate(outside, size=(540, 960), mode='nearest')
        outside = outside[0][0].numpy()
        mask[np.where(outside == 1)] = 12

        return mask


@torch.no_grad()
def slide_inference(images, model, num_classes=13, crop_size=(1024, 1024), stride=(768, 768)):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = images.size()

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = images.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = images.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = images[:, :, y1:y2, x1:x2]

            crop_seg_logit = model(pixel_values=crop_img)[0]
            crop_seg_logit = F.interpolate(
                crop_seg_logit,
                size=crop_size,
                mode="bilinear",
                align_corners=False
            )
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2),
                            int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0
    seg_logits = preds / count_mat

    return seg_logits


@torch.no_grad()
def _slide_inference(images, model, num_classes=13, crop_size=(1024, 1024), stride=(768, 768)):
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    batch_size, _, h_img, w_img = images.size()

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = images.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = images.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = images[:, :, y1:y2, x1:x2]

            crop_seg_logit = model(pixel_values=crop_img)[0]
            crop_seg_logit = F.interpolate(
                crop_seg_logit,
                size=crop_size,
                mode="bilinear",
                align_corners=False
            )
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2),
                            int(y1),
                            int(preds.shape[2] - y2)))

            count_mat[:, :, y1:y2, x1:x2] += 1
    assert (count_mat == 0).sum() == 0

    return preds, count_mat
