import math
import os
import itertools
from glob import glob

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

from segformers.losses import FocalLoss, AuxMixLoss
from segformers.utils import compute_mIoU


class Trainer:
    def __init__(
        self,
        model,
        config,
    ):
        
        # GPU가 2개 이상인 경우 DataParallel 사용
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            print(f"Number of GPUs: {num_gpus}. Using DataParallel.")
            self.multi_run = True
            self._model = nn.DataParallel(model)

        else:
            print(f"Number of GPU: {num_gpus}. Not using DataParallel.")
            self.multi_run = False
            self._model = model

        self.n_epochs = config['n_epochs']
        self.dir_ckpt = config['dir_ckpt']
 
        param_optimizer = {
            'segformer': list(self.model.segformer.named_parameters()),
            'domain_classifier': list(self.model.domain_classifier.named_parameters())
        }
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer['segformer'] if not any(nd in n for nd in no_decay)],
            'lr': config['optimizer']['lr'],
            'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer['segformer'] if any(nd in n for nd in no_decay)],
            'lr': config['optimizer']['lr'],
            'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer['domain_classifier'] if not any(nd in n for nd in no_decay)],
            'lr': config['optimizer_D']['lr'],
            'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer['domain_classifier'] if any(nd in n for nd in no_decay)],
            'lr': config['optimizer_D']['lr'],
            'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)
                               #, **config['optimizer'])
        self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, **config['scheduler'])
        # self.criterion = FocalLoss()
        self.threshold_scheduler = DynamicThreshold(start_threshold=0.9, end_threshold=0.7, total_epochs=self.n_epochs)
        self.lambda_scheduler = LambdaScheduler(max_epochs=self.n_epochs)
        self.adjuster = None
        #self.adjuster = PerformanceBasedAdjuster(initial_alpha=0.9, adjustment_rate=0.05)

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = FocalLoss()
        self.criterion_D = nn.BCEWithLogitsLoss()
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
        self.model.to(self.device)
        self.best_metric = 0.0
        wandb.init(**config['wandb'], config=config)

    @property
    def model(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module
        else:
            return self._model

    def fit(self, train_loader, valid_loader, target_loader):

        for e in range(self.n_epochs):
            self.lambda_scheduler.current_epoch = e
            self.threshold_scheduler.current_epoch = e
            train_scores = self.train(train_loader, target_loader)
            valid_scores = self.evaluate(valid_loader)
            
            if self.adjuster:
                self.adjuster.update(valid_scores['mIoU'])
            else:
                pass
            
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

    def train(self, source_loader, target_loader):
        self.model.train()
        n = 0
        scores = {'Loss': 0.0,
                  'Semantic Loss': 0.0,
                  'Domain Loss': 0.0,
                  'mIoU': 0.0}
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        
        for _ in range(max(len(source_loader), len(target_loader))):
            self.optimizer.zero_grad()
            # train G
            for params in self.model.domain_classifier.parameters():
                params.requires_grad = False

            # train with target
            try:
                target_data = next(target_iter)
                target_data = target_data['pixel_values'].float().to(self.device)
            except StopIteration:
                target_iter = iter(target_loader)
                target_data = next(target_iter)
                target_data = target_data['pixel_values'].float().to(self.device)

            lamda = self.lambda_scheduler.get_lambda()
            target_semantic_outputs, target_domain_outputs = self.model(target_data, lamda=lamda)

            # pseudo labeling
            pseudo_labels = torch.argmax(target_semantic_outputs, dim=1).long()  # get the predicted class labels

            prediction_probs = F.softmax(target_semantic_outputs, dim=1)  # get the softmax probabilities
            max_probs, _ = torch.max(prediction_probs, dim=1)  # get the max probability for each prediction

            threshold = self.threshold_scheduler.get_threshold()
            mask = max_probs.ge(threshold).float()  # create a mask for predictions with confidence >= threshold

            # Multiply the pseudo labels with mask so that we only consider high confidence predictions.
            pseudo_labels_masked = mask * pseudo_labels.float()
            pseudo_labels_masked = pseudo_labels_masked.long()
            pseudo_labels_masked = pseudo_labels_masked.detach() # backpropagation에 영향을 주지 않도록하기 위해

            target_semantic_loss = self.criterion(target_semantic_outputs, pseudo_labels_masked)
            # train with source
            try:
                source_items = next(source_iter)
                source_data, source_labels = source_items['pixel_values'].float().to(self.device), source_items['labels'].long().to(self.device)
            except StopIteration:
                source_iter = iter(source_loader)
                source_items = next(source_iter)
                source_data, source_labels = source_items['pixel_values'].float().to(self.device), source_items['labels'].long().to(self.device)
            
            source_semantic_outputs, source_domain_outputs = self.model(source_data)
            source_semantic_outputs = upsampled_logit(source_semantic_outputs, source_labels)
            source_semantic_loss = self.criterion(source_semantic_outputs, source_labels)

            semantic_loss = source_semantic_loss + target_semantic_loss

            # train D
            for params in self.model.domain_classifier.parameters():
                params.requires_grad = True

            # train with source
            source_domain_loss = self.criterion_D(source_domain_outputs, torch.zeros_like(source_domain_outputs))

            # train with target
            target_domain_loss = self.criterion_D(target_domain_outputs, torch.ones_like(target_domain_outputs))

            domain_loss = source_domain_loss + target_domain_loss

            # Convex combination
            if self.adjuster:
                alpha = self.adjuster.alpha
                beta = self.adjuster.beta
            else:
                alpha = 0.5
                beta = 0.5
            total_loss = alpha * semantic_loss + beta * domain_loss
            total_loss.backward()
            self.optimizer.step()

            batch_size = len(source_data)
            n += batch_size
            _, predictions = torch.max(source_semantic_outputs, 1)
            scores['Loss'] += batch_size * total_loss.item()
            scores['Semantic Loss'] += batch_size * semantic_loss.item()
            scores['Domain Loss'] += batch_size * domain_loss.item()
            for pred, gt in zip(predictions, source_labels):
                scores['mIoU'] += compute_mIoU(pred, gt)

        for k, v in scores.items():
            scores[k] = v / n

        return scores

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        n = 0
        scores = {'Semantic Loss': 0.0, 'mIoU': 0.0}
        for inputs in loader:
            images, masks = inputs['pixel_values'], inputs['labels']
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            
            semantic_outputs, _ = self.model(images)  # 도메인 출력은 무시
            semantic_outputs = upsampled_logit(semantic_outputs, masks)
            
            _, predictions = torch.max(semantic_outputs, 1) 

            semantic_loss = self.criterion(semantic_outputs, masks)  # semantic loss만 계산


            batch_size = len(images)
            n += batch_size
            scores['Semantic Loss'] += batch_size * semantic_loss.item()
            # 배치 내에 있는 샘플들에 대해 mIoU를 계산
            for pred, gt in zip(predictions, masks):
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

class LambdaScheduler:
    """
    The domain adaptation parameter λ is initiated at 0 and is gradually changed to 1 using the following schedule
        lambda(t) = 2 / (1 + exp(-gamma * p)) - 1

    References)
    [1] Yaroslav Ganin et al, Domain-Adversarial Training of Neural Networks, Conference: The Journal of Machine Learning Research (JMLR), 2016

    """
    def __init__(self, max_epochs, alpha=10):
        self.max_epochs = max_epochs
        self.current_epoch = 1
        self.alpha = alpha

    def get_lambda(self):
        p = self.current_epoch / self.max_epochs
        return 2.0 / (1. + np.exp(-self.alpha * p)) - 1.
    

class DynamicThreshold:
    def __init__(self, start_threshold, end_threshold, total_epochs):
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.total_epochs = total_epochs
        self.delta = (start_threshold - end_threshold) / total_epochs
        self.current_epoch = None

    def get_threshold(self):
        return max(self.start_threshold - self.current_epoch * self.delta, self.end_threshold)


class PerformanceBasedAdjuster:
    def __init__(self, initial_alpha, adjustment_rate):
        self.alpha = initial_alpha
        self.beta = 1 - initial_alpha
        self.adjustment_rate = adjustment_rate
        self.best_metric = float('-inf')

    def update(self, current_metric):
        if current_metric > self.best_metric:
            self.best_metric = current_metric
        else:
            self.alpha = max(0, self.alpha - self.adjustment_rate)
            self.beta = 1 - self.alpha
            

def upsampled_logit(logits, masks):
    return nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
    
