import os
from datetime import datetime

from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from segformers.config import config_dann
from dann.DomainAdaptation import SegFormerDANN
from segformers.datasets import SourceDataset, TargetDataset
from segformers.networks import SegFormer
from segformers.trainer_dann import Trainer
from segformers.transforms import augmentation, augmentation_base, transform_base
from segformers.utils import seed_all, print_env

config = config_dann

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # Set an experiment
    print_env()
    seed_all(config['seed'])
    run_id = int(datetime.timestamp(datetime.now()))
    config['run_id'] = run_id
    config['dir_ckpt'] = os.path.join(config['dir_ckpt'], str(run_id))
    os.makedirs(config['dir_ckpt'])
    model = SegFormerDANN(segformer=SegFormer)
    image_processor = SegformerImageProcessor(
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    train_dataset = SourceDataset(
        root=config['dir_data'],
        csv_file='full.csv',
        image_processor=image_processor,
        transform=augmentation,
        is_training=True
    )

    valid_dataset = SourceDataset(
        root=config['dir_data'],
        csv_file='full.csv',
        image_processor=image_processor,
        transform=None,
        is_training=True
    )

    target_dataset = TargetDataset(
        root=config['dir_data'],
        csv_file='train_target.csv',
        image_processor=image_processor,
        transform=None,
        is_training=True
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=1, shuffle=True, num_workers=4)

    trainer = Trainer(model, config)
    trainer.fit(train_loader=train_loader,
                valid_loader=valid_loader,
                target_loader=target_loader)
