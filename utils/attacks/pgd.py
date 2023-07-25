import torch
from torchattacks import PGD
from utils.multitask_losses import compute_mtloss


from utils.gradients import record_metric_gradients


class MTPGD(PGD):
    def __init__(self, model, tasks, criteria, eps=0.3, alpha=2 / 255, steps=40, random_start=False,
                 record_all_tasks=False, algorithm="PGD"):
        super(MTPGD, self).__init__(model, eps, alpha, steps, random_start)

        self.tasks = tasks
        self.algorithm = algorithm
        self.criteria =  {k : l for (k,l) in criteria.items() if k in self.tasks}
        self.record_all_tasks = record_all_tasks
        self.reset_acc()

    def reset_acc(self):
        self.robust_accs = {}

    def append_acc(self, vals, lbls=None):
        vals = dict(zip(lbls,vals)) if lbls is not None else vals
        self.robust_accs = {k:self.robust_accs.get(k,[])+[v] for (k,v) in vals.items()}



    def forward(self, images, labels, threshold=0.5):
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
        images = images.clone().detach().to(self.device)
        if isinstance(labels, dict):
            labels = {k : l.clone().detach().to(self.device) for (k,l) in labels.items() if k !="rep"}

        outputs_clean = self.model(images)
        if "TRADES" in self.algorithm:
            labels = {k:outputs_clean[k] for k in labels.keys()}

        evaluate_tasks = list(labels.keys())
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for i in range(self.steps):
            if self.record_all_tasks:
                adv_images_metrics = adv_images.clone().detach()
                self.gradients, self.labels, self.gradients_cov, self.gradients_cosine, self.gradients_dot, self.gradients_magn, self.gradients_curve = \
                    record_metric_gradients(self.model, self._targeted, adv_images_metrics, labels, self.gradients,
                                            self.labels, self.gradients_cov,
                                            self.gradients_cosine, self.gradients_dot, self.gradients_magn,
                                            self.gradients_curve)

            adv_images.requires_grad_()
            with torch.enable_grad():
                outputs = self.model(adv_images)

                if not isinstance(outputs, dict):
                    outputs = {f"class_object#{k}":outputs[:,i] for (i,k) in enumerate(self.model.pathologies)}

                loss, loss_dict, avg_losses = compute_mtloss(self.criteria, outputs, labels, equally=True,
                                                             loss_dict={}, avg_losses=None, algorithm=self.algorithm)

                cost = -loss if self._targeted else loss

                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]


            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        outputs = self.model(adv_images)

        #if len(outputs) == len(tasks):
        #    outputs

        if isinstance(outputs, dict):
            is_hot_encoded = len(list(outputs.values())[0].shape) > 1
        else:
            outputs = {f'class_object#{k}':outputs[:,i] for (i,k) in enumerate(self.model.pathologies)}
            is_hot_encoded = False

        outputs_labels = list(outputs.keys())

        if "class_object#multilabel" in outputs_labels:
            acc_labels = [e for e in outputs_labels if e in evaluate_tasks]
            robust_acc= torch.stack([torch.argmax(outputs[k],1) == torch.argmax(labels[k].squeeze(),1) if is_hot_encoded else (outputs[k]>threshold).long() == (labels[k]>threshold).long() for k in acc_labels if len(labels[k].shape)<3])
            self.append_acc(robust_acc.sum(1)/len(images), acc_labels)


        return adv_images



class MLPGD(PGD):
    def __init__(self, model, tasks, criteria, eps=0.3, alpha=2 / 255, steps=40, random_start=False, record_all_tasks=False):
        super(MLPGD, self).__init__(model, eps, alpha, steps, random_start)

        self.tasks = tasks
        self.criteria =  criteria
        self.record_all_tasks = record_all_tasks
        self.reset_acc()

    def reset_acc(self):
        self.robust_accs = {}

    def append_acc(self, vals, lbls=None):
        vals = dict(zip(lbls,vals)) if lbls is not None else vals
        self.robust_accs = {k:self.robust_accs.get(k,[])+[v] for (k,v) in vals.items()}

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        #labels = {l:lbl.clone().detach().to(self.device) for (l,lbl) in labels.items()} if isinstance(labels,dict) else labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = self.criteria

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        outputs = self.model(adv_images)
        robust_acc = torch.stack(
            [torch.argmax(outputs[k], 1) == torch.argmax(labels[k].squeeze(), 1) for k in labels.keys()])
        self.append_acc(robust_acc.sum(1) / len(images), list(labels.keys()))
        return adv_images