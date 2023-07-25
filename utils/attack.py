from torchattacks.attacks.pgd import PGD
from NIH_Chest_X_Rays.losses import FocalLoss
import torch
class PGDL(PGD):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFAULT: 0.3)
        alpha (float): step size. (DEFAULT: 2/255)
        steps (int): number of steps. (DEFAULT: 40)
        random_start (bool): using random initialization of delta. (DEFAULT: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False, min_value=-1024, max_value=1024):

        self.min_value = min_value
        self.max_value= max_value
        #eps = min_value + (max_value-min_value) * eps
        #alpha = min_value + (max_value-min_value) * alpha

        super(PGDL, self).__init__(model, eps, alpha, steps, random_start)



    def forward(self, images, labels, loss=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)

        if loss is None or loss=="ce":
            loss = torch.nn.CrossEntropyLoss().to(self.device)
            labels = labels.long()
        elif loss=="focal":
            loss = FocalLoss(gamma = 2, device=self.device)
        elif loss=="bce":
            loss = torch.nn.BCEWithLogitsLoss().to(self.device)


        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            random_noise = (self.max_value-self.min_value) * torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = adv_images +random_noise
            adv_images = torch.clamp(adv_images, min=self.min_value, max=self.max_value).detach()

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            alpha =  (self.max_value-self.min_value) * self.alpha
            adv_images = adv_images.detach() - alpha * grad.sign()
            eps = self.min_value + (self.max_value-self.min_value) * self.eps
            neg_eps = self.min_value - (self.max_value-self.min_value) * self.eps
            delta = torch.clamp(adv_images - images, min=neg_eps, max=eps)
            adv_images = torch.clamp(images + delta, min=self.min_value, max=self.max_value).detach()

        return adv_images

