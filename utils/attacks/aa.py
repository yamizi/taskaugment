import torch
from torch import nn
from torchattacks import AutoAttack, MultiAttack, APGD, APGDT, FAB, Square
from utils.multitask_losses import compute_mtloss

class MultitaskToMono(nn.Module):
    def __init__(self, model, task):
        super(MultitaskToMono, self).__init__()
        self.model = model
        self.task = task

    def forward(self, x):
        y = self.model(x)
        return y[self.task]

class MTAA(AutoAttack) :
    def __init__(self, model, tasks, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False):
        super().__init__(model, norm, eps, version, n_classes=n_classes, seed=seed, verbose=verbose)
        self.tasks = tasks
        task = "class_object#{}".format(self.tasks[0])
        model = MultitaskToMono(model, task)
        if version == 'standard':
            self.autoattack = MultiAttack([
                APGD(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, loss='ce', n_restarts=1),
                APGDT(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                FAB(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_classes=n_classes, n_restarts=1),
                Square(model, eps=eps, norm=norm, seed=self.get_seed(), verbose=verbose, n_queries=5000, n_restarts=1),
            ])

        else:
            raise ValueError("Not valid version. ['standard']")


    def forward(self, images, labels):
        r"""
        Overridden.
        """

        self.gradients = []
        self.gradients_cov = []
        self.gradients_cosine = []
        self.gradients_dot = []
        self.gradients_magn = []
        self.gradients_curve = []

        self.labels = []

        task = "class_object#{}".format(self.tasks[0])
        images = images.clone().detach().to(self.device)
        labels = labels[task].clone().detach().to(self.device)
        adv_images = self.autoattack(images, labels.argmax(1))

        return adv_images
