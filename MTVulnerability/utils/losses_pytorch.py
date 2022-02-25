import sys, os
sys.path.append("./midlevel-reps")
from visualpriors.taskonomy_network import TaskonomyDecoder

import torch

import torch.nn.functional as F
import torch.nn as nn

SMOOTH = 1e-6
CHANNELS_TO_TASKS = {
    1: ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', ],
    2: ['curvature', 'principal_curvature'],
    3: ['autoencoding', 'denoising', 'normal', 'inpainting', 'rgb', 'normals'],
    128: ['segment_unsup2d', 'segment_unsup25d'],
    1000: ['class_object'],
    None: ['segment_semantic']
}

TASKS_TO_CHANNELS = {}
for n, tasks in CHANNELS_TO_TASKS.items():
    for task in tasks:
        TASKS_TO_CHANNELS[task] = n

PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal', 'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'segment_semantic', ]
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS


def heteroscedastic_normal(mean_and_scales, target, weight=None, eps=1e-2):
    mu, scales = mean_and_scales
    loss = (mu - target)**2 / (scales**2 + eps) + torch.log(scales**2 + eps)
#     return torch.sum(weight * loss) / torch.sum(weight) if weight is not None else loss.mean()
    return torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean()

def heteroscedastic_double_exponential(mean_and_scales, target, weight=None, eps=5e-2):
    mu, scales = mean_and_scales
    loss = torch.abs(mu - target) / (scales + eps) + torch.log(2.0 * (scales + eps))
    return torch.mean(weight * loss) / weight.mean() if weight is not None else loss.mean()


def iou_loss(outputs: torch.Tensor, labels: torch.Tensor, threshold:float=None):

    outputs, labels = torch.argmax(outputs, axis=1), torch.argmax(labels, axis=1)

    outputs = outputs.squeeze(1)  if len(outputs.shape)>3 else outputs # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    if threshold is not None:
        return torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou  # IOU returns 1 when error is null (similar iou) and 0 when totally different



def weighted_mse_loss(inputs, target, weight=None):
    if weight is not None:
#         sq = (inputs - target) ** 2
#         weightsq = torch.sum(weight * sq)
        return torch.mean(weight * (inputs - target) ** 2)/torch.mean(weight)
    else:
        return F.mse_loss(inputs, target)

def weighted_l1_loss(inputs, target, weight=None):
    if weight is not None:
        return torch.mean(weight * torch.abs(inputs - target))/torch.mean(weight)
    return F.l1_loss(inputs, target)

def perceptual_l1_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['inputs_decoded'] = inputs_decoded
        cache['targets_decoded'] = targets_decoded

        if weight is not None:
            return torch.mean(weight * torch.abs(inputs_decoded - targets_decoded))/torch.mean(weight)
        return F.l1_loss(inputs_decoded, targets_decoded)
    return runner


def perceptual_cross_entropy_loss(decoder_path, bake_decodings):
    task = [t for t in SINGLE_IMAGE_TASKS if t in decoder_path][0]
    decoder = TaskonomyDecoder(TASKS_TO_CHANNELS[task], feed_forward=task in FEED_FORWARD_TASKS)
    checkpoint = torch.load(decoder_path)
    decoder.load_state_dict(checkpoint['state_dict'])
    decoder.cuda()
    decoder.eval()
    print(f'Loaded decoder from {decoder_path} for perceptual loss')
    def runner(inputs, target, weight=None, cache={}):
        # the last arguments are so we can 'cache' and pass the decodings outside
        inputs_decoded = decoder(inputs)
        targets_decoded = target if bake_decodings else decoder(target)
        cache['inputs_decoded'] = inputs_decoded
        cache['targets_decoded'] = targets_decoded

        batch_size, _ = targets_decoded.shape
        return -1. * torch.sum(torch.softmax(targets_decoded, dim=1) * F.log_softmax(inputs_decoded, dim=1)) / batch_size
    return runner


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """

        if target.dim()==4:
            target = torch.argmax(target, axis=1)

        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss
