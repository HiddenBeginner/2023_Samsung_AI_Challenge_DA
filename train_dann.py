import os
from datetime import datetime

from torch.utils.data import DataLoader
from transformers import SegformerImageProcessor

from dann.config import config
from dann.DomainAdaptation import DANN, FullyConvolutionalDiscriminator
from dann.DomainAdaptation import get_backbone, get_classifier
from dann.backbones import DeepLabV3, LRASPP
from dann.datasets import SourceDataset, TargetDataset
from segformers.networks import SegFormer
from dann.trainer import Trainer
from segformers.transforms import augmentation, augmentation_base, transform_base
from segformers.utils import seed_all

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Set an experiment
    seed_all(config['seed'])
    run_id = int(datetime.timestamp(datetime.now()))
    config['run_id'] = run_id
    config['dir_ckpt'] = os.path.join(config['dir_ckpt'], str(run_id))
    os.makedirs(config['dir_ckpt'])
    backbone = DeepLabV3(backbone='resnet50', pretrained=True, mode='all')
    classifiers = get_classifier(backbone, aux=True)
    model = DANN(
        feature_backbone=get_backbone(backbone), 
        semantic_classifier=classifiers[0], 
        aux_classifier=classifiers[1],
        domain_classifier=FullyConvolutionalDiscriminator(num_classes=13))
    # image_processor = SegformerImageProcessor(
    #     image_mean=[0.485, 0.456, 0.406],
    #     image_std=[0.229, 0.224, 0.225],
    # )

    train_dataset = SourceDataset(
        root=config['dir_data'],
        csv_file='full.csv',
#        image_processor=image_processor,
        transform=augmentation_base,
        is_training=True
    )

    valid_dataset = SourceDataset(
        root=config['dir_data'],
        csv_file='full.csv',
#        image_processor=image_processor,
        transform=transform_base,
        is_training=True
    )

    target_dataset = TargetDataset(
        root=config['dir_data'],
        csv_file='train_target.csv',
        #image_processor=image_processor,
        transform=transform_base,
        is_training=True
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=4, shuffle=True, num_workers=4)

    trainer = Trainer(model, config)
    trainer.fit(train_loader=train_loader,
                valid_loader=valid_loader,
                target_loader=target_loader)
