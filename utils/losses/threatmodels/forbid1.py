import torch

from utils.losses import TopkSortedLossSampled, OrdinalLoss

def get_LOSSES(loss):
    from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
    losses = {"mse":MSELoss, "ce":CrossEntropyLoss, "bce":BCEWithLogitsLoss}

    return losses.get(loss, MSELoss)


def ForbidTopClasses(reduction="mean", criterion=None, experiment=None, log_sum=False, direct=False):

    include_ce = criterion.find("_ce")>-1
    use_ordinal = criterion.find("ord")>-1
    kwargs = {}
    if use_ordinal:
        success_loss= OrdinalLoss()
        kwargs = {"include_ce":include_ce}
    elif criterion.find("sortedsampled")>-1:
        success_loss = TopkSortedLossSampled(reduction,criterion,experiment=experiment)
        kwargs = {"log_sum": log_sum,"include_ce":include_ce}

    else:
        loss =get_LOSSES(criterion)
        success_loss = loss(reduction=reduction)


    def loss_direct(y_pred, y_target):
        # kwargs: include_ce=True, use_exp=False
        target = torch.ones_like(y_pred)

        for i,t in enumerate(target):
            t[y_target[i].long()] = 0
            target[i] = t

        return success_loss(y_pred, target, **kwargs)

    def loss_indirect(y_pred, y_target):
        # kwargs: include_ce=True, use_exp=False
        target = torch.zeros_like(y_pred)

        for i,t in enumerate(target):
            t[y_target[i].long()] = 1
            target[i] = t

        return -success_loss(y_pred, target, **kwargs)

    return loss_direct if direct=="asc" else loss_indirect

