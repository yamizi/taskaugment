import torch
import torch.nn.functional as F
import numpy as np
from utils.losses import TopkSortedLossSampled, OrdinalLoss

def get_LOSSES(loss):
    from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss,BCELoss
    losses = {"mse":MSELoss, "ce":CrossEntropyLoss, "bce":BCEWithLogitsLoss, "bcel":BCELoss}
    return losses.get(loss, MSELoss)


def SortAllClasses(reduction="mean", criterion=None, experiment=None, log_sum=False, direct=False):
    from utils.losses import SortABClasses

    include_ce = criterion.find("_ce")>-1
    use_ordinal = criterion.find("ord")>-1
    sortab_loss = SortABClasses(reduction=reduction, criterion=criterion,experiment=experiment, direct=direct)

    kwargs = {}
    if use_ordinal:
        success_loss= OrdinalLoss()
        kwargs = {"include_ce":include_ce}
    elif criterion.find("sortedsampled")>-1:
        success_loss = TopkSortedLossSampled(reduction,criterion,experiment=experiment)
        kwargs = {"log_sum": log_sum,"include_ce":include_ce}

    else:
        loss = get_LOSSES(criterion)
        success_loss = loss(reduction=reduction)


    def loss_direct(y_pred, y_target):
        # kwargs: include_ce=True, use_exp=False

        if y_pred.shape == y_target.shape:
            target = y_target
        else:
            nb_classes = y_pred.shape[1]
            nb_targets = y_target.shape[1]
            target = torch.zeros_like(y_pred).to(y_pred.device)
            for i in range(nb_targets):
                labels = F.one_hot(y_target[:,i], num_classes=nb_classes)
                target = torch.masked_fill(target, labels.bool(),(i+1)*1/nb_targets)

        ls = success_loss(y_pred.float(), target.float(), **kwargs)
        return ls


    def loss_indirect(y_pred, y_target):
        # kwargs: include_ce=True, use_exp=False

        if y_pred.shape == y_target.shape:
            y_target= y_target.argsort(1)
        l= 0
        for i in range(y_target.shape[1]):
            for j in range(y_target.shape[1]):
                if i>=j:
                    continue

                target = torch.cat([y_target[:,i].unsqueeze(1),y_target[:,j].unsqueeze(1)],1)
                l = l +sortab_loss(y_pred, target)

        return l


    return loss_direct if direct=="direct" else loss_indirect
