# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import json
import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
from loss import FocalLoss
import cv2



# Function to save a batch of images as a grid
def save_batch_as_grid(batch, output_file):
    # Ensure the batch is on the CPU and convert to NumPy
    batch = batch.cpu().numpy()  # Convert from tensor to NumPy array
    batch = np.transpose(batch, (0, 2, 3, 1))  # Convert from (N, C, H, W) to (N, H, W, C)
    batch = (batch * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8 if needed

    # Calculate grid size (e.g., 4x8 for 32 images)
    grid_rows = int(np.ceil(np.sqrt(batch.shape[0])))
    grid_cols = int(np.ceil(batch.shape[0] / grid_rows))

    # Create a blank canvas for the grid
    img_height, img_width = batch.shape[1], batch.shape[2]
    grid_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)

    # Place each image in the grid
    for idx, img in enumerate(batch):
        row = idx // grid_cols
        col = idx % grid_cols
        grid_image[row * img_height:(row + 1) * img_height, col * img_width:(col + 1) * img_width, :] = img

    # Save the grid image as a PNG file
    cv2.imwrite(output_file, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
    print(f"Batch saved as grid image at: {output_file}")



def misc_measure(confusion_matrix, y_true, y_pred_proba):
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score = []
    confusion_matrices = []
    roc_auc = None

    for i in range(1, confusion_matrix.shape[0]):
        cm1 = confusion_matrix[i]
        confusion_matrices.append(cm1)  # Store the confusion matrix

        total = np.sum(cm1)
        tp = cm1[1, 1]
        tn = cm1[0, 0]
        fp = cm1[0, 1]
        fn = cm1[1, 0]

        # Accuracy
        acc.append((tp + tn) / total if total != 0 else 0)

        # Sensitivity
        sensitivity_ = tp / (tp + fn) if (tp + fn) != 0 else 0
        sensitivity.append(sensitivity_)

        # Specificity
        specificity_ = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity.append(specificity_)

        # Precision
        precision_ = tp / (tp + fp) if (tp + fp) != 0 else 0
        precision.append(precision_)

        # Geometric Mean (G)
        G.append(np.sqrt(sensitivity_ * specificity_))

        # F1 Score
        F1_score.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_) if (precision_ + sensitivity_) != 0 else 0)

    # Calculate ROC AUC
    if y_true is not None and y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = None  # Handle cases where ROC AUC cannot be computed

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "geometric_mean": G,
        "f1_score": F1_score,
        "confusion_matrices": confusion_matrices,
        "roc_auc": roc_auc
    }





def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
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

    for data_iter_step, (data, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if args.use_metadata:
            samples = data[0].to(device, non_blocking=True)
            metadata = data[1].to(device, non_blocking=True)
        else : 
            samples = data.to(device, non_blocking=True)

        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if args.use_metadata:
                outputs = model(samples, metadata)
            else : 
                outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    

@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, criterion, num_class, use_metadata, output_dir):
    # criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        if use_metadata:
            images = batch[0][0]
            metadata = batch[0][1].to(device, non_blocking=True)
        else : 
            images = batch[0]
        
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        # with torch.cuda.amp.autocast():
        output = model(images, metadata) if use_metadata else model(images)
        loss = criterion(output, target)
        prediction_softmax = nn.Softmax(dim=1)(output)
        _,prediction_decode = torch.max(prediction_softmax, 1)
        _,true_label_decode = torch.max(true_label, 1)

        prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
        true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
        true_label_onehot_list.extend(true_label.cpu().detach().numpy())
        prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list, labels=[i for i in range(num_class)])
    # acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    # auc_roc = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovo', average='macro')
    # auc_pr = average_precision_score(true_label_onehot_list, prediction_list, average='macro')          
            
    metric_logger.synchronize_between_processes()

    # Extract metrics from the results of misc_measure
    results = misc_measure(confusion_matrix, true_label_onehot_list, prediction_list)
    acc = results["accuracy"]
    sensitivity = results["sensitivity"]
    specificity = results["specificity"]
    precision = results["precision"]
    auc_roc = results["roc_auc"]
    F1 = results["f1_score"]
    geometric_mean = results["geometric_mean"]

    print('Metrics - Acc: {:.4f} AUC-roc: {:.4f} F1-score: {:.4f}'.format(
        np.mean(acc), auc_roc, np.mean(F1)
    )) 

    results_path = os.path.join(output_dir, task + '_metrics_{}.csv'.format(mode))
    with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2_name = [["acc", "sensitivity", "specificity", "precision", "auc_roc", "F1", "geometric_mean", "loss"]]
        data2 = [[
            np.mean(acc), 
            np.mean(sensitivity), 
            np.mean(specificity), 
            np.mean(precision), 
            auc_roc, 
            np.mean(F1), 
            np.mean(geometric_mean), 
            metric_logger.loss
        ]]
        for name, i in zip(data2_name, data2):
            wf.writerow(name)
            wf.writerow(i)
            

    logits_path = os.path.join(output_dir, task + '_logits_{}.csv'.format(mode))
    pred_truth = {"truth" : [[int(i[1])] for i in true_label_onehot_list], "predictions" : [[float(i[1])] for i in prediction_list]}
    with open(logits_path, 'w') as fp:
        json.dump(pred_truth, fp)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=False,plot_lib="matplotlib")
        plt.savefig(os.path.join(output_dir, task+'confusion_matrix_test.jpg'),dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, sensitivity, specificity, F1



@torch.no_grad()
def evaluate_classifier(data_loader, model, device, task, epoch, mode, num_class, use_metadata, output_dir):
    
    # MODEL MUST BE ON CPU NOT DEVICE
    
    assert use_metadata==True
    
    # Initialize metrics
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    # Switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        metadata = batch[0]
        target = batch[-1]
        true_label = torch.nn.functional.one_hot(target.to(torch.int64), num_classes=num_class)

        # Compute predictions
        with torch.cuda.amp.autocast():
            prediction = model(metadata)  # Model handles threshold internally

        prediction_list.extend(prediction.cpu().detach().numpy())
        true_label_decode_list.extend(target.cpu().detach().numpy())
        true_label_onehot_list.extend(true_label.cpu().detach().numpy())

    # Convert to NumPy arrays for metric calculations
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_list = np.array(prediction_list, dtype=int)

    # Compute confusion matrix
    confusion_matrix = multilabel_confusion_matrix(
        true_label_decode_list, prediction_list, labels=[i for i in range(num_class)]
    )

    # Compute evaluation metrics
    results = misc_measure(confusion_matrix)
    acc = results["accuracy"]
    sensitivity = results["sensitivity"]
    specificity = results["specificity"]
    precision = results["precision"]
    auc_roc = results["roc_auc"]
    F1 = results["f1_score"]
    geometric_mean = results["geometric_mean"]
    # auc_roc = roc_auc_score(true_label_onehot_list, prediction_list, multi_class='ovo', average='macro')
    # auc_pr = average_precision_score(true_label_onehot_list, prediction_list, average='macro')
    auc_roc = 0.0
    auc_pr = 0.0

    metric_logger.synchronize_between_processes()

    # Log metrics
    print('Metrics - Acc: {:.4f} AUC-roc: {:.4f} F1-score: {:.4f}'.format(
        np.mean(acc), auc_roc, np.mean(F1)
    ))

    # Save results
    results_path = os.path.join(output_dir, task + '_metrics_{}.csv'.format(mode))
    with open(results_path, mode='a', newline='', encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2_name = [["acc", "sensitivity", "specificity", "precision", "auc_roc", "F1", "geometric_mean", "loss"]]
        data2 = [[
            np.mean(acc),
            np.mean(sensitivity),
            np.mean(specificity),
            np.mean(precision),
            auc_roc,
            np.mean(F1),
            np.mean(geometric_mean),
            metric_logger.loss
        ]]
        for name, i in zip(data2_name, data2):
            wf.writerow(name)
            wf.writerow(i)

    logits_path = os.path.join(output_dir, task + '_logits_{}.csv'.format(mode))
    pred_truth = {"truth" : [[int(i[0])] for i in true_label_onehot_list], "predictions" : [[int(i)] for i in prediction_list]}
    with open(logits_path, 'w') as fp:
        json.dump(pred_truth, fp)

    # Generate confusion matrix plot for test mode
    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_list)
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=False, plot_lib="matplotlib")
        plt.savefig(os.path.join(output_dir, task + '_confusion_matrix_test.jpg'), dpi=600, bbox_inches='tight')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, auc_roc, sensitivity, specificity, F1


