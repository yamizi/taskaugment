import torch

def MultiLabelCrossEntropyLoss(reduction="mean"):

    def multilabel_categorical_crossentropy(y_pred, y_true):

        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        if reduction =="mean":
            return (neg_loss + pos_loss).mean()
        else:
            return neg_loss + pos_loss


    return  multilabel_categorical_crossentropy


def MultiLabelHingeLoss(reduction="mean"):

    def multilabel_hinge(y_pred, y_true):

        loss = y_true * (1-y_pred) + (1-y_true) * y_pred
        if reduction =="mean":
            return loss.mean()
        else:
            return loss


    return multilabel_hinge


def LInfLoss(reduction="mean"):

    def loss_fn(y_pred, y_true):
        import numpy as np
        loss =  torch.cdist(y_true,y_pred, np.inf)
        return loss.mean() if reduction == "mean" else loss
