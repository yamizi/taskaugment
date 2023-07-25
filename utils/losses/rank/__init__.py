import torch
import numpy as np
from differentiable_sorting.torch import bitonic_matrices, diff_argsort

def RankLoss(reduction="mean"):

    def rank(y_pred, y_target_binary):

        y_target_binary = y_target_binary.flatten()
        y_pred = y_pred.flatten()
        pos_args = y_target_binary.round()==1
        neg_args = y_target_binary.round() == 0

        loss =  torch.max(torch.exp(y_pred[neg_args])) - torch.min(torch.exp(y_pred[pos_args]))
        return torch.clamp(loss,min=0)

    return rank


def SortedLoss(reduction="mean", cpu=True):

    def loss(y_pred, y_target):

        power_size = np.ceil(np.log(len(y_target))/np.log(2))
        missing = int(2**power_size - len(y_target))
        y_target = torch.cat([y_target,torch.zeros(missing).cuda()])
        y_pred = torch.cat([y_pred, torch.zeros(missing).cuda()])

        if cpu:
            y_pred = y_pred.cpu()
            y_target = y_target.cpu()

        sort_matrices = bitonic_matrices(len(y_pred),y_pred.device)
        sort_outs =diff_argsort(sort_matrices, -y_pred)
        sort_lbls = diff_argsort(sort_matrices, -y_target)
        return torch.nn.MSELoss(reduction=reduction)(sort_outs.float(), sort_lbls.float())


    return loss
