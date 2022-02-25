import numpy as np
from utils.losses_pytorch import weighted_mse_loss, iou_loss, CrossEntropy2d
import torch
from torch.autograd import Variable
from learning.utils_learn import clamp_tensor

def get_torch_std(info):
    std_array = np.asarray(info["std"])
    tensor_std = torch.from_numpy(std_array)
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    return tensor_std

def PGD_attack_mtask(x, y, mask, net, criterion, task_name, epsilon, steps, dataset, step_size, info, args, using_noise=True):
    net.eval()
    if epsilon == 0:
        return Variable(x, requires_grad=False)

    GPU_flag = False
    if torch.cuda.is_available():
        GPU_flag=True

    rescale_term = 2./255
    epsilon = epsilon * rescale_term
    step_size = step_size * rescale_term

    x_adv = x.clone()

    pert_upper = x_adv + epsilon
    pert_lower = x_adv - epsilon

    upper_bound = torch.ones_like(x_adv)
    lower_bound = -torch.ones_like(x_adv)

    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    ones_x = torch.ones_like(x).float()
    if GPU_flag:

        x_adv = x_adv.cuda()
        upper_bound = upper_bound.cuda()
        lower_bound = lower_bound.cuda()
        for keys, m in mask.items():
            mask[keys] = m.cuda()
        for keys, tar in y.items():
            y[keys] = tar.cuda()


    if using_noise:
        noise = torch.FloatTensor(x.size()).uniform_(-epsilon, epsilon)
        if GPU_flag:
            noise = noise.cuda()
        x_adv = x_adv + noise
        x_adv = clamp_tensor(x_adv, lower_bound, upper_bound)

    x_adv = Variable(x_adv, requires_grad=True)

    for i in range(steps):
        h_adv = net(x_adv)

        grad_total_loss = None
        for each in task_name:
            if grad_total_loss is None:
                grad_total_loss = criterion[each](h_adv[each], y[each], mask[each])
            else:
                grad_total_loss = grad_total_loss + criterion[each](h_adv[each], y[each], mask[each])

        net.zero_grad()

        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)

        grad_total_loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv + step_size * x_adv.grad
        x_adv = clamp_tensor(x_adv, upper_bound, lower_bound)

        x_adv = Variable(x_adv.data, requires_grad=True)

    # sample =x_adv.data
    # im_rgb = np.moveaxis(sample[1].cpu().numpy().squeeze(), 0, 2)
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(im_rgb)
    # plt.show()

    return x_adv


class MultiOutputClassifier(object):
    def __init__(self, *args, **kwargs):

        self.multi_task = False
        self.dense_task = False
        self._nb_tasks = 1

        classes = kwargs.get("nb_classes", None)

        if classes is None:
            classes = self._output

        assert classes is not None

        if isinstance(classes, list):
            self.multi_task = True
            self._nb_tasks = len(classes)

        if any([len(c)==3 for c in classes]):
            self.dense_task = True

        self._nb_classes = classes

    @property
    def nb_tasks(self):
        return self._nb_tasks


def compute_success_dense(classifier: "Classifier",y: np.ndarray,target: np.ndarray,
    targeted: bool = False, metric:str="IOU"):

    metric = "CE" if metric is None else metric

    if metric =="MSE":
        score = weighted_mse_loss(torch.Tensor(y),torch.Tensor(target))
        score = 1 / score if targeted else score

    elif metric =="IOU":
        y1,y2 =torch.Tensor(y), torch.Tensor(target)
        score = iou_loss(y1,y2)
        score = score if targeted else 1- score

    elif metric == "CE":
        loss = CrossEntropy2d()
        y1, y2 = torch.Tensor(y), torch.Tensor(target)
        score = loss(y1, torch.argmax(y2, axis=1))
        score = 1- score if targeted else score

    return score.numpy()

def compute_success_class(classifier: "Classifier",y: np.ndarray,target: np.ndarray,targeted: bool = False):
    return np.argmax(y, axis=1) == np.argmax(target, axis=1) \
        if targeted else np.argmax(y, axis=1) != np.argmax(target, axis=1)

def compute_success_multitask(classifier: "Classifier",y: np.ndarray,target: np.ndarray,
    targeted: bool = False, metric:str="MSE"):

    success = [compute_success_dense(classifier, y[i], t, targeted, metric) if classifier.dense_task else \
            compute_success_class(classifier, y[i], t, targeted) for i,t in enumerate(target)]

    return success

def compute_success_array(
    classifier: "Classifier",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
    metric: str="IOU"
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    """
    y_adv = classifier.predict(x_adv, batch_size=batch_size)
    y_clean = labels if targeted else classifier.predict(x_clean, batch_size=batch_size)

    if classifier.multi_task:
        attack_success = compute_success_multitask(classifier, y_adv, y_clean, metric)

    else:
        attack_success = compute_success_dense(classifier, y_adv, y_clean, targeted, metric) if classifier.dense_task else \
            compute_success_class(classifier, y_adv, y_clean, targeted)

    attack_success = np.array(attack_success)

    return attack_success


def compute_success(
    classifier: "Classifier",
    x_clean: np.ndarray,
    labels: np.ndarray,
    x_adv: np.ndarray,
    targeted: bool = False,
    batch_size: int = 1,
    strategy: str = "MEAN",
    metric: str=None
) -> float:
    """
    Compute the success rate of an attack based on clean samples, adversarial samples and targets or correct labels.

    :param classifier: Classifier used for prediction.
    :param x_clean: Original clean samples.
    :param labels: Correct labels of `x_clean` if the attack is untargeted, or target labels of the attack otherwise.
    :param x_adv: Adversarial samples to be evaluated.
    :param targeted: `True` if the attack is targeted. In that case, `labels` are treated as target classes instead of
           correct labels of the clean samples.
    :param batch_size: Batch size.
    :return: Percentage of successful adversarial samples.
    :rtype: `float`
    """
    attack_success = compute_success_array(classifier, x_clean, labels, x_adv, targeted, batch_size, metric=metric)

    if not classifier.multi_task:
        return np.sum(attack_success) / x_adv.shape[0]

    if strategy == "MEAN":
        return np.sum(attack_success) / (2*x_adv.shape[0])
    elif strategy.isnumeric():
        np.sum(attack_success[int(strategy)]) / x_adv.shape[0]
    else:
        return np.sum(attack_success, axis=1) / x_adv.shape[0]
