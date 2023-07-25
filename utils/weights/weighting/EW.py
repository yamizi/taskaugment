import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .abstract_weighting import AbsWeighting

class EW(AbsWeighting):
    r"""Equally Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` means the number of tasks.

    """
    def __init__(self):
        super(EW, self).__init__()
        
    def backward(self, losses, **kwargs):
        losses = torch.cat([e.unsqueeze(0) for e in losses]) if isinstance(losses, list) else losses
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        loss.backward()
        return np.ones(self.task_num)