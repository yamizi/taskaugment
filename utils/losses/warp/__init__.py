from .loss import WARPLoss


def WarpLoss(reduction="mean"):
    wl = WARPLoss()

    def loss(y_pred, y_true):

        y_target = (y_true>0).float()
        l =  wl(y_pred,y_target)
        return l

    return loss
