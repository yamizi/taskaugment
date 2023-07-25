#Adapted from https://github.com/columbia/MTRobust

import torch
import math

def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image

def back_transform(image, info):
    # image = image2.copy()

    image[:, 0, :, :] = image[:, 0, :, :] * info["std"][0] + info["mean"][0]
    image[:, 1, :, :] = image[:, 1, :, :] * info["std"][1] + info["mean"][1]
    image[:, 2, :, :] = image[:, 2, :, :] * info["std"][2] + info["mean"][2]
    return image

def forward_transform(image, info):
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image

class Houdini(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Y_pred, Y, task_loss, ignore_index=255):

        normal_dist     = torch.distributions.Normal(0.0, 1.0)
        probs           = 1.0 - normal_dist.cdf(Y_pred - Y)
        loss            = torch.sum(probs * task_loss.squeeze()) #* mask.squeeze(1)) / torch.sum(mask.float())

        ctx.save_for_backward(Y_pred, Y, task_loss)
        return loss

    @staticmethod
    def backward(ctx, grad_output):

        Y_pred, Y, task_loss = ctx.saved_tensors

        C = 1./math.sqrt(2 * math.pi)

        grad_input  = C * torch.exp(-1.0 * (torch.abs(Y - Y_pred) ** 2) / 2.0) * task_loss.squeeze()

        return (grad_output * grad_input, None, None, None)
