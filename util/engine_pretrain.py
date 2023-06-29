# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
from util.misc import adjust_learning_rate

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    lr: int, total_epochs: int, warmup_epochs: int):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, lr, total_epochs, warmup_epochs)
        samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=0.8)
        loss_value = loss.item()
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        metric_logger.update(loss=loss_value)
        lr_updated = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr_updated)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    losses = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[-1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples)
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae1'].update(loss.item(), n=batch_size)
        losses.append(loss.cpu().numpy())
    # gather the stats from all processes
    print('* mae1@1 {top1.global_avg:.3f} loss(MSE) {losses.global_avg:.3f}'.format(top1=metric_logger.mae1, losses=metric_logger.loss))
    return losses


def train_one_epoch_finetune(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, lr: int, total_epochs: int, warmup_epochs: int):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, lr, total_epochs, warmup_epochs)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets.float())
        loss_value = loss.item()
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        metric_logger.update(loss=loss_value)
        lr_updated = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr_updated)
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_finetune(data_loader, model, device):
    criterion = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    outputs_vector = []
    targets_vector = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[-1]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            outputs = outputs.squeeze()
            loss = criterion(outputs, targets)
        mae1 = L1Loss(outputs, targets)
        outputs_vector.append(outputs.cpu().numpy())
        targets_vector.append(targets.cpu().numpy())
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae1'].update(mae1.item(), n=batch_size)
    # gather the stats from all processes
    print('* mae1@1 {top1.global_avg:.3f} loss(MSE) {losses.global_avg:.3f}'
          .format(top1=metric_logger.mae1, losses=metric_logger.loss))
    return outputs_vector, targets_vector