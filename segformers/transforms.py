import albumentations as A
import numpy as np
import torch
from albumentations.core.transforms_interface import DualTransform
from albumentations.pytorch import ToTensorV2

def fisheye_circular_transform_torch(image, mask=None, fov_degree=200, focal_scale=4.5):
    img = image if image is not None else mask
    _, h, w = img.shape

    # Convert degrees to radians using torch tensor
    radian_conversion = torch.tensor(np.pi/180, dtype=img.dtype, device=img.device)

    # Calculate the focal length using the given FOV
    f = w / (2 * torch.tan(0.5 * fov_degree * radian_conversion))
    f_scaled = f * focal_scale

    # Meshgrid for coordinates
    x = torch.linspace(-w//2, w//2, w).repeat(h, 1)
    y = torch.linspace(-h//2, h//2, h).unsqueeze(1).repeat(1, w)
    r = torch.sqrt(x*x + y*y)
    theta = torch.atan2(y, x)

    # Apply fisheye transformation
    r_fisheye = f_scaled * torch.atan(r / f_scaled)
    x_fisheye = (w // 2 + r_fisheye * torch.cos(theta)).long()
    y_fisheye = (h // 2 + r_fisheye * torch.sin(theta)).long()

    # Create masks for valid coordinates
    valid_coords = (x_fisheye >= 0) & (x_fisheye < w) & (y_fisheye >= 0) & (y_fisheye < h) & (r < w // 2)

    # Initialize output images
    new_image = None
    new_mask = None

    if image is not None:
        new_image = torch.zeros_like(image)
        new_image[:, valid_coords] = image[:, y_fisheye[valid_coords], x_fisheye[valid_coords]]

    if mask is not None:
        new_mask = torch.zeros_like(mask) + 12
        new_mask[:, valid_coords] = mask[:, y_fisheye[valid_coords], x_fisheye[valid_coords]]

    return new_image, new_mask


class FisheyeTransform(DualTransform):
    def __init__(self, fov_degree=200, focal_scale=4.5, always_apply=False, p=1.0):
        super(FisheyeTransform, self).__init__(always_apply, p)
        self.fov_degree = fov_degree
        self.focal_scale = focal_scale

    def apply(self, image, **params):
        image_tensor = torch.tensor(image).permute(2, 0, 1).float()
        transformed_image, _ = fisheye_circular_transform_torch(
            image_tensor, None, fov_degree=self.fov_degree, focal_scale=self.focal_scale
            )
        return transformed_image.permute(1, 2, 0).byte().numpy()

    def apply_to_mask(self, mask, **params):
        mask_tensor = torch.tensor(mask).unsqueeze(0).float()
        _, transformed_mask = fisheye_circular_transform_torch(
            None, mask_tensor, fov_degree=self.fov_degree, focal_scale=self.focal_scale
            )
        return transformed_mask.squeeze(0).byte().numpy()


augmentation = A.Compose(
    [
        #FisheyeTransform(p=0.75),
        A.RandomScale(scale_limit=(-0.5, 0.0), p=1.0),
        A.RandomCrop(512, 512, p=1.0),
        A.HorizontalFlip(),
        A.OneOf([
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.HueSaturationValue(p=1.0)
        ])
    ]
)

augmentation_pl = A.Compose(
    [
        FisheyeTransform(p=0.75),
        A.RandomScale(scale_limit=(-0.5, 0.0), p=1.0),
        A.RandomCrop(512, 512, p=1.0),
    ]
)


augmentation_base = A.Compose([
    FisheyeTransform(p=0.5),
    A.RandomCrop(width=224, height=224),
    A.RandomScale(scale_limit=0.2, p=0.2),
    #A.RandomRotate90(p=0.5),
    #A.HorizontalFlip(p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
    A.ColorJitter(p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    #A.CoarseDropout(max_holes=8, max_height=16, max_width=16, fill_value=0, p=0.5),
    A.Resize(513, 513),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

transform_base = A.Compose(
    [   
        A.Resize(513, 513),
        A.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet 데이터의 통계량으로 정규화
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
)