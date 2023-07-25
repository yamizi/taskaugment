import torch
from torch import nn
import numpy as np
from utils.losses import Houdini
from torch.nn import functional as F

def TopkLoss(reduction="mean", criterion=None, experiment=None):
    """
    Use a combination of the two losses
    loss1 tries to move the correct top k labels to the highest probabilities
    loss2 tries to move down the top k predicted labels to their correct probabilities

    :param reduction:
    :param criterion: "max" or a nn.Loss or None
    :param experiment: Comet Experiment
    :return:
    """
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target, use_exp=False):
        k = (y_target[0] > 0).sum().item()

        arg_target = torch.argsort(-y_target)[:, :k]
        val_targets = y_target[:, arg_target]
        val_pred = y_pred[:, arg_target]

        val_targets = torch.exp(val_targets.float()) if use_exp else val_targets.float()
        val_pred = torch.exp(val_pred.float()) if use_exp else val_pred.float()
        loss1 = torch.max(val_targets - val_pred) if criterion == "max" else criterion(val_targets, val_pred)

        arg_pred = torch.argsort(-y_pred)[:, :k]
        val_targets = y_target[:, arg_pred]
        val_pred = y_pred[:, arg_pred]

        val_targets = torch.exp(val_targets.float()) if use_exp else val_targets.float()
        val_pred = torch.exp(val_pred.float()) if use_exp else val_pred.float()
        loss2 = torch.max(val_targets - val_pred) if criterion == "max" else criterion(val_targets, val_pred)

        l = loss1 + loss2

        if experiment is not None:
            experiment.log_metric("loss1", loss1)
            experiment.log_metric("loss2", loss2)

        return l

    return loss


def TopkSortedLoss(reduction="mean", criterion=None, experiment=None):
    if criterion is None:
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target, log_sum=False):
        ref_target = y_target[0]
        nb_labels = len(ref_target)
        alpha = 0
        l = 0

        for k in range(nb_labels):
            mask_j = ref_target < ref_target[k]

            for j in range(nb_labels):
                if mask_j[j]:
                    if log_sum:
                        l = l + torch.exp(criterion(y_pred[:, j] ,y_pred[:, k]))
                    else:
                        l = l + torch.max(torch.Tensor([0, alpha + y_pred[:, j] - y_pred[:, k]]))

        if experiment is not None:
            experiment.log_metric("loss4", l)

        if log_sum:
            l = torch.log(1 + l)
        return l

    return loss


def TopkSortedLossSampled(reduction="mean", criterion=None, random_sample=250, experiment=None):
    if criterion is None or isinstance(criterion, str):
        criterion = nn.MSELoss(reduction=reduction)

    def loss(y_pred, y_target, log_sum=False, include_ce=True):
        ref_labels = y_target[0]
        ref_targets = torch.nonzero(ref_labels)
        nb_labels = ref_labels.shape[0]
        alpha = 0.05
        l = 0

        for k in ref_targets:
            mask_j = ref_labels < ref_labels[k.item()]
            counter = random_sample
            while counter > 0:
                j = np.random.choice(range(nb_labels), 1)
                if mask_j[j]:
                    if log_sum:
                        l = l + torch.exp(criterion(y_pred[:, j.item()], y_pred[:, k.item()]))
                    else:
                        l = l + torch.max(
                            torch.Tensor([0, alpha + criterion(y_pred[:, j.item()], y_pred[:, k.item()])]))
                    counter = counter - 1

        if experiment is not None:
            experiment.log_metric("loss4_sampled", l)

        if log_sum:
            l = torch.log(1 + l)
        l = Houdini.apply(y_pred, y_target, l)

        if include_ce:
            return F.binary_cross_entropy(y_pred, y_target) + l if len(y_target.shape)==1 else  F.binary_cross_entropy_with_logits(y_pred, y_target) + l

        else:
            return l

    return loss
