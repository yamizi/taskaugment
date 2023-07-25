import torch

from utils.losses import TopkSortedLossSampled, OrdinalLoss

def get_LOSSES(loss):
    from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
    losses = {"mse":MSELoss, "ce":CrossEntropyLoss, "bce":BCEWithLogitsLoss}
    return losses.get(loss, MSELoss)


def SortABClasses(reduction="mean", criterion=None, experiment=None, log_sum=False, direct=False):

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
        loss = get_LOSSES(criterion)
        success_loss = loss(reduction=reduction)


    def loss(y_pred, y_target):
        # kwargs: include_ce=True, use_exp=False
        target_a = y_target[:,0]
        target_b = y_target[:, 1]

        prob_a = y_pred.gather(1, target_a.expand((1, -1)).transpose(0, 1))
        prob_b = y_pred.gather(1, target_b.expand((1, -1)).transpose(0, 1))

        pred = torch.cat([prob_a,prob_b],1)
        target = torch.cat([torch.zeros_like(prob_a),torch.ones_like(prob_b)],1)

        l =  success_loss(pred, target, **kwargs)
        return l


    return loss
