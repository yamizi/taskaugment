import torch
import collections
from torch.nn.modules.module import Module


class MaxupCrossEntropyLoss(Module):
    def __init__(self, m, batch_size):
        super().__init__()
        self.m = m
        self.batch_size = batch_size

    def forward(self, input, target):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        input = input.reshape((self.batch_size, -1))
        target = target.reshape((self.batch_size, -1))
        loss = torch.sum(-target * logsoftmax(input), dim=1)
        loss, _ = loss.reshape((self.batch_size//self.m, self.m)).max(1)
        loss = torch.mean(loss)
        return loss

def cross_entropy_loss_mask(output, target, mask=None):
    if mask is None:
        raise TypeError("Mask is None")
    else:
        out =  torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1),
                                                 reduction='none')
        if len(mask.shape) == 4: mask = mask.squeeze(1)
        out *= mask.float()
        return out.mean()



def cross_entropy_loss(output, target, mask=None):
    return torch.nn.functional.cross_entropy(output.float(), target.long().squeeze(dim=1),
                                             reduction='mean')

def soft_cross_entropy_loss(output, target, mask=None):
    log_likelihood = -torch.nn.functional.log_softmax(output.float(), dim=-1)
    return torch.mean(torch.mul(log_likelihood, target.float()))

def TRADES_loss(output, target, mask=None):
    if mask is not None:
        output = output[mask].squeeze().reshape(output.shape)
        target = target[mask].squeeze().reshape(output.shape)
    log_likelihood_output = torch.nn.functional.log_softmax(output.float(),  dim=1)
    log_likelihood_target = torch.nn.functional.log_softmax(target.float(), dim=1)

    return torch.nn.KLDivLoss(reduction='sum',log_target=True)(log_likelihood_output.float(), log_likelihood_target.float())

def l1_loss(output, target, mask=None):
    return torch.nn.functional.l1_loss(output, target, reduction='mean')

def l1_loss_mask(output, target, mask=None):
    if mask is None:
        raise TypeError("Mask is None")
    else:
        out = torch.nn.functional.l1_loss(output, target, reduction='none')
        out *= mask.float()
        return out.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_mtloss(criteria,output, target, equally=True, loss_dict={},avg_losses={}, algorithm="PGD"):
    sum_loss = None

    for c_name, criterion_fun in criteria.items():
        try:
            task_output = output[c_name].float().squeeze() if output.get(c_name) is not None else output['class_object#'+c_name].float().squeeze()
            task_label = target[c_name].float().squeeze() if target.get(c_name) is not None else target['class_object#'+c_name].float().squeeze()

            if(task_output.shape !=task_label.shape and isinstance(criterion_fun, torch.nn.BCEWithLogitsLoss)):
                task_output = task_output.squeeze(1)

            mask = ~torch.isnan(task_label)
            if "TRADES" in algorithm:
                this_loss = TRADES_loss(task_output, task_label, mask)
            else:
                this_loss = criterion_fun(task_output[mask], task_label[mask])
        except Exception as e:
            print("!!! exception: ", e)

        if equally:
            this_loss = this_loss * 1.0 / len(criteria.keys())

        if sum_loss is None:
            sum_loss = this_loss
        else:
            sum_loss = sum_loss + this_loss

        loss_dict[c_name] = this_loss
        if avg_losses is not None:
            avg_losses[c_name].update(loss_dict[c_name].data.item(), output["rep"].size(0))

    loss = sum_loss

    return loss, loss_dict, avg_losses

def get_criteria(task_set=None):
    criteria = {}

    tasks = []
    loss_map = {
        'autoencoder'           : l1_loss,
        'gabor': l1_loss,
        'hog': l1_loss,
        'sift': l1_loss,
        'class_object'          : soft_cross_entropy_loss,
        'class_places'          : soft_cross_entropy_loss,
        'depth_euclidean'       : l1_loss_mask,
        'depth_zbuffer'         : l1_loss_mask,
        'depth'         : l1_loss,
        'edge_occlusion'        : l1_loss_mask,
        'edge_texture'          : l1_loss,
        'keypoints2d'           : l1_loss,
        'keypoints3d'           : l1_loss_mask,
        'normal'                : l1_loss_mask,
        'principal_curvature'   : l1_loss_mask,
        'reshading'             : l1_loss_mask,
        'room_layout'           : soft_cross_entropy_loss,
        'segment_unsup25d'      : l1_loss,
        'segment_unsup2d'       : l1_loss,
        'segmentsemantic'       : cross_entropy_loss_mask,
        'segment_semantic'       : cross_entropy_loss_mask,
        'vanishing_point'       : l1_loss,
        'sex': soft_cross_entropy_loss,
        'age': l1_loss,
        'jigsaw': soft_cross_entropy_loss,
        'rotation': soft_cross_entropy_loss,
        'autoencoder1c':l1_loss

    }

    if task_set is None:
        return loss_map, list(loss_map.keys())

    for task in task_set:
        base_task = task.split("#")[0]
        if base_task in loss_map:
            criteria[task]  = loss_map[base_task]
            tasks.append(task)
        else:
            print('unknown classes')


    return criteria, tasks

