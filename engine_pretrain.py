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
import util.lr_sched as lr_sched

import cv2
import numpy as np

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        
        # Save a sample image for visualization
        if data_iter_step == 0:  # Save only the first batch of the epoch
            batch_images = samples.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to NHWC format
            batch_images = (batch_images - batch_images.min()) / (batch_images.max() - batch_images.min())  # Normalize to [0, 1]
            batch_images = (batch_images * 255).astype('uint8')  # Scale to [0, 255] and convert to uint8
            
            # Create a grid of 32 images (assuming 4x8 grid)
            grid_rows, grid_cols = 4, 8
            img_height, img_width = batch_images.shape[1], batch_images.shape[2]
            grid_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype='uint8')
            
            for idx, img in enumerate(batch_images[:grid_rows * grid_cols]):
                row = idx // grid_cols
                col = idx % grid_cols
                grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img
                
                save_path = '/data/mkondrac/foundation_model_cardio/code/RETFound_MAE/visualize_data_augmnetation_pretrain/sample_grid.png'
                cv2.imwrite(save_path, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        
        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}