from .houdini import Houdini
from .warp import WarpLoss

from .sorted import TopkLoss, TopkSortedLoss, TopkSortedLossSampled
from .multilabel import LInfLoss, MultiLabelHingeLoss, MultiLabelCrossEntropyLoss
from .ordinal import OrdinalLoss

from .threatmodels import ForbidTopClasses, ForceTopClasses, SortABClasses, SortAllClasses

from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss

def co_occurence_loss(output, co_occurence):
    loss = output * co_occurence
    return loss.mean()
