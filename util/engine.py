"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
from .losses import build_target, dice_loss
from util import utils as utils
import torch.nn as nn
from util.metrics import Metrics


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        dice_target = build_target(target, num_classes, ignore_index)
        loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    losses['out'] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    set_training_mode=True,
                    print_freq=0,
                    writer=None,
                    args=None):
    """
        Train the model for one epoch.

        Args:
            model (torch.nn.Module): The model to be trained.
            data_loader (Iterable): The data loader for the training data.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            device (torch.device): The device used for training (CPU or GPU).
            epoch (int): The current training epoch.
            loss_scaler: The object used for gradient scaling.
            clip_grad (float, optional): The maximum value for gradient clipping. Default is 0, which means no gradient clipping.
            clip_mode (str, optional): The mode for gradient clipping, can be 'norm' or 'value'. Default is 'norm'.
            set_training_mode (bool, optional): Whether to set the model to training mode. Default is True.
            set_bn_eval (bool, optional): Whether to set the batch normalization layers to evaluation mode. Default is False.
            print_freq (int): The frequent of printing info
            writer (Optional[Any], optional): The object used for writing TensorBoard logs.
            args (Optional[Any], optional): Additional arguments.

        Returns:
            Dict[str, float]: A dictionary containing the average values of the training metrics.
    """

    model.train(set_training_mode)

    num_steps = len(data_loader)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if args.nb_classes == 2:
        # TODO set CrossEntropy loss-weights for object & background according to your dataset
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for idx, (img, lbl) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        img = img.to(device)
        lbl = lbl.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(img)
            loss = criterion(logits, lbl, loss_weight, num_classes=args.nb_classes, ignore_index=args.ignore_index)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        with torch.cuda.amp.autocast():
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                        parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(loss=loss_value, lr=lr)

        if idx % print_freq == 0:
            if args.local_rank == 0:
                iter_all_count = epoch * num_steps + idx
                writer.add_scalar('train_loss', loss, iter_all_count)
                # writer.add_scalar('grad_norm', grad_norm, iter_all_count)
                writer.add_scalar('train_lr', lr, iter_all_count)

    metric_logger.synchronize_between_processes()
    torch.cuda.empty_cache()

    return metric_logger.meters["loss"].global_avg, lr


@torch.inference_mode()
def evaluate(data_loader: Iterable,
             model: torch.nn.Module,
             device: torch.device,
             print_freq,
             writer, args):
    """
        Evaluate the model for one epoch.

        Args:
            data_loader (Iterable): The data loader for the valid data.
            model (torch.nn.Module): The model to be evaluated.
            device (torch.device): The device used for training (CPU or GPU).
            epoch (int): The current training epoch.
            print_freq (int): The frequent of printing info
            writer (Optional[Any], optional): The object used for writing TensorBoard logs.
            args (Optional[Any], optional): Additional arguments.
            visualization (bool, optional): Whether to use TensorBoard visualization. Default is True.

        Returns:
            Dict[str, float]: A dictionary containing the average values of the training metrics.
    """

    model.eval()

    metric = Metrics(args.nb_classes, args.ignore_label, args.device)
    confmat = utils.ConfusionMatrix(args.nb_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for idx, (images, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # compute output
        # with torch.cuda.amp.autocast(): # TODO: ConfusionMatrix not implemented for 'Half' data
        outputs = model(images)
        confmat.update(labels.flatten(), outputs.argmax(1).flatten())
        metric.update(outputs, labels.flatten())

        if writer:
            if idx % print_freq == 0 & args.local_rank == 0:
                writer.add_scalar('valid_mf1', metric.compute_f1()[1])
                writer.add_scalar('valid_acc', metric.compute_pixel_acc()[1])
                writer.add_scalar('valid_mIOU', metric.compute_iou()[1])

    confmat.reduce_from_all_processes()
    metric.reduce_from_all_processes()

    torch.cuda.empty_cache()

    return confmat, metric