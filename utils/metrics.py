import torch
import numpy as np
import sklearn, sklearn.metrics
from skimage.metrics import structural_similarity as ssim

import torchxrayvision as xrv


def common_similarity_metrics(experiment, img, img2, raw=False):
    ssim_m = np.array([ssim(image.cpu().numpy(), img2[i].cpu().numpy(), data_range=image.max() - image.min()) for i, image in enumerate(img)])
    l2 = torch.cdist(img, img2,2)
    linf = torch.cdist(img, img2,np.inf)

    if not raw:
        linf = linf.mean()
        l2 = l2.mean()
        ssim_m =ssim_m.mean()

    if experiment is not None:
        experiment.log_metric("linf", linf)
        experiment.log_metric("l2", l2)
        experiment.log_metric("ssim", ssim_m)

    return ssim_m, l2, linf


def common_success_metrics(experiment, clean_output, adv_output, labels, target_outputs=None):
    aucs = compute_auc(labels, clean_output, log=False)
    adv_aucs = compute_auc(labels, adv_output, log=False)
    original_acc = topkaccuracy(adv_output, clean_output, (1, 3, 5))
    if target_outputs is not None:
        target_acc = topkaccuracy(adv_output, target_outputs, (1, 3, 5))

    if experiment is not None:
        experiment.log_metric("clean_auc", aucs)
        experiment.log_metric("adv_auc", adv_aucs)

        if target_outputs is not None:
            experiment.log_metric("target_top1_acc", target_acc[0])
            experiment.log_metric("target_top3_acc", target_acc[1])
            experiment.log_metric("target_top5_acc", target_acc[2])

        experiment.log_metric("original_top1_acc", original_acc[0])
        experiment.log_metric("original_top3_acc", original_acc[1])
        experiment.log_metric("original_top5_acc", original_acc[2])

        from utils.losses import MSELoss, BCEWithLogitsLoss, MultiLabelCrossEntropyLoss, co_occurence_loss
        loss_ = MSELoss(reduction="mean")
        mse = loss_(adv_output, clean_output)
        experiment.log_metric("mse", mse)

        loss_ = BCEWithLogitsLoss(reduction="mean")
        bce = loss_(adv_output, clean_output)
        experiment.log_metric("bce", bce)

        loss_ = MultiLabelCrossEntropyLoss(reduction="mean")
        mlce = loss_((adv_output > 0.5).float(), (clean_output > 0.5).float())
        experiment.log_metric("mlce", mlce)

        mlacc = ((adv_output > 0.5) == (clean_output > 0.5)).float().mean()
        experiment.log_metric("mlacc", mlacc)

        if target_outputs is not None:
            mlacc_target = ((adv_output > 0.5) == (target_outputs > 0.5)).float().mean()
            experiment.log_metric("mlacc_target", mlacc_target)

            co_loss = co_occurence_loss(adv_output, target_outputs)
            experiment.log_metric("co_loss", co_loss)


def compute_auc(labels,outputs, log=True):
    labs = labels.cpu().detach()
    outs = outputs.cpu().detach()
    aucs = []
    for i in range(14):
        if len(np.unique(np.asarray(labs)[:, i])) > 1:
            auc = sklearn.metrics.roc_auc_score(np.asarray(labs)[:, i], np.asarray(outs)[:, i])
        else:
            auc = np.nan
        if log:
            print(xrv.datasets.default_pathologies[i], auc)
        aucs.append(auc)
    return auc

def topkaccuracy(output, target, topk=(1,), return_mean=True):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.shape)>1:
            target = target.argmax(1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float()
            if return_mean:
                res.append(correct_k.sum().mul_(100.0 / batch_size).item())
            else:
                res.append(correct_k.sum(0).cpu().numpy())
        return res

def success_forbid(output, target, topk=(1,), return_mean=True):
    """Computes the % of absence over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if len(target.shape) > 1:
            target = target.argmax(1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = ~correct[:k].any(0, keepdim=True)
            if return_mean:
                res.append(correct_k.float().sum().mul_(100.0 / batch_size).item())
            else:
                res.append(correct_k.float().sum(0).cpu().numpy())
        return res


def success_sort_ab(output, target_a,target_b, return_mean=True):
    """Computes the % of inputs where 2 values are sorted a<b"""
    with torch.no_grad():
        batch_size = output.size(0)

        prob_a = output.gather(1,target_a.expand((1,-1)).transpose(0,1)).squeeze(1)
        prob_b = output.gather(1,target_b.expand((1,-1)).transpose(0,1)).squeeze(1)

        correct = prob_a<=prob_b

        success = correct.float()

        if return_mean:
            return success.sum().mul_(100.0 / batch_size).item()
        else:
            return success.cpu().numpy()



def success_force_all(output, targets, topk=(1,), return_mean=True):
    batch_size = output.size(0)
    top_vals = [np.ones(batch_size) for k in topk]
    for target in targets:
        top_val = topkaccuracy(output, target, topk, return_mean=False)
        vals = [top_vals[i]*e for i,e in enumerate(top_val)]
        top_vals = vals

    top_vals = np.array(top_vals)
    return top_vals.sum(1)/batch_size*100 if return_mean else top_vals


def success_sort_all(output, targets, return_mean=True):
    batch_size = output.size(0)
    top_vals = np.ones(batch_size)
    for i, target_a in enumerate(targets):
        for j, target_b in enumerate(targets):
            if i >= j:
                continue

            top_val = success_sort_ab(output, target_a, target_b, return_mean=False)
            top_vals = top_vals * top_val

    return top_vals.sum()/batch_size*100 if return_mean else top_vals




