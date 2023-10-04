import math
import os
from glob import glob

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

import wandb

from .losses import DiceLoss, label_to_one_hot_label
from .utils import compute_mIoU


class Trainer:
    def __init__(
        self,
        model,
        config,
    ):
        self.model = model
        self.dice = config['criterion']['dice']
        self.n_epochs = config['n_epochs']
        self.dir_ckpt = config['dir_ckpt']

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, **config['optimizer'])

        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, **config['scheduler'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = DiceLoss()
        self.model.to(self.device)
        self.best_metric = 0.0
        wandb.init(**config['wandb'], config=config)

    def fit(self, train_loader, valid_loader):
        for e in range(self.n_epochs):
            train_scores = self.train(train_loader)
            valid_scores = self.evaluate(valid_loader)

            log = {'Epoch': e + 1, 'LR': self.scheduler.get_lr()[0]}
            for k, v in train_scores.items():
                log[f'train_{k}'] = v

            for k, v in valid_scores.items():
                log[f'valid_{k}'] = v

            msg = ''
            for k, v in log.items():
                msg += f'{k}: {v:.4f} | '
            print(msg[:-1])
            wandb.log(log)

            self.save(f'{self.dir_ckpt}/last_ckpt.bin')
            if valid_scores['mIoU'] > self.best_metric:
                self.best_metric = valid_scores['mIoU']
                self.save(f'{self.dir_ckpt}/best_ckpt_{str(e+1).zfill(4)}.bin')
                # Keep top 3 models
                for path in sorted(glob(f'{self.dir_ckpt}/best_ckpt_*.bin'))[:-3]:
                    os.remove(path)

            self.scheduler.step()
        wandb.finish()

    def train(self, loader):
        self.model.train()
        n = 0
        scores = {'Loss': 0.0, 'mIoU': 0.0}
        for inputs in loader:
            images, masks = inputs['pixel_values'], inputs['labels']
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            _, logits = self.model(pixel_values=images, labels=masks)

            self.optimizer.zero_grad()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            loss1 = self.criterion1(upsampled_logits, masks)
            if self.dice:
                labels = label_to_one_hot_label(masks, 13, self.device)
                loss2 = self.criterion2(upsampled_logits, labels)
                loss = loss1 + loss2
            else:
                loss = loss1

            loss.backward()
            self.optimizer.step()
            _, predicted = upsampled_logits.max(1)

            batch_size = len(images)
            n += batch_size
            scores['Loss'] += batch_size * loss.item()
            for pred, gt in zip(predicted, masks):
                scores['mIoU'] += compute_mIoU(pred, gt)

        for k, v in scores.items():
            scores[k] = v / n

        return scores

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        n = 0
        scores = {'Loss': 0.0, 'mIoU': 0.0}
        for inputs in loader:
            images, masks = inputs['pixel_values'], inputs['labels']
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            _, logits = self.model(pixel_values=images, labels=masks)

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
            loss1 = self.criterion1(upsampled_logits, masks)
            if self.dice:
                labels = label_to_one_hot_label(masks, 13, self.device)
                loss2 = self.criterion2(upsampled_logits, labels)
                loss = loss1 + loss2
            else:
                loss = loss1

            _, predicted = upsampled_logits.max(1)

            batch_size = len(images)
            n += batch_size
            scores['Loss'] += batch_size * loss.item()
            # 배치 내에 있는 샘플들에 대해 mIoU를 계산
            for pred, gt in zip(predicted, masks):
                scores['mIoU'] += compute_mIoU(pred, gt)

        for k, v in scores.items():
            scores[k] = v / n

        return scores

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    """
    https://gaussian37.github.io/dl-pytorch-lr_scheduler/
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) *
                    (1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
