
import torch as t
from torch.nn import Parameter as P
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F

"""
From: https://github.com/missingdaysqxy/pytorch_OrdinalNet/blob/master/core/ordinalloss.py
"""

def ordinal_loss(input: t.Tensor, target: t.Tensor, MaxValue):
    if (t.argsort(input) == t.argsort(target)).all():
        return 0
    else:
        in_padL = F.pad(input, [1, 0], mode='constant', value=input[-1].data)
        in_padR = F.pad(input, [0, 1], mode='constant', value=input[0].data)
        in_diff = in_padR - in_padL
        tar_padL = F.pad(target, [1, 0], mode='constant', value=target[-1].data)
        tar_padR = F.pad(target, [0, 1], mode='constant', value=target[0].data)
        tar_diff = tar_padR - tar_padL
        loss = F.mse_loss(in_diff / MaxValue, tar_diff / MaxValue)
        return loss


class OrdinalLoss(_WeightedLoss):
    def __init__(self, MaxValue=1, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(OrdinalLoss, self).__init__(weight, size_average, reduce, reduction)
        assert MaxValue != 0
        self.MaxVaule = MaxValue
        self.w = P(t.Tensor(2))

    def forward(self, input: t.Tensor, target: t.Tensor, include_ce=True):
        l = 0
        for i,trg in enumerate(target):
            l = l+ordinal_loss(F.softmax(input[i],0), F.softmax(trg,0), self.MaxVaule)

        l = l/len(target)
        if include_ce:
            return F.binary_cross_entropy(input, target) + l if len(target.shape)==1 else  F.binary_cross_entropy_with_logits(input, target) + l

        else:
            return l

