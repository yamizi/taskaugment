
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Union

import numpy as np

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)

from utils.art.attacks.numpy_PGD import MultiObjectiveProjectedGradientDescentNumpy
from utils.art.attacks.pytorch_PGD import MultiObjectiveProjectedGradientDescentPytorch

import logging
logger = logging.getLogger(__name__)

class MultiObjectiveProjectedGradientDescent(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """


    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
        "max_iter",
        "random_eps",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(
        self,
        estimator,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        strategy:str = "WEIGHTED_TASKS",
        strategy_params: dict = {},
        strategy_success:str = "MEAN",
        project:bool = True
    ):
        """
        Create a :class:`.ProjectedGradientDescent` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this
                           method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        """
        super(MultiObjectiveProjectedGradientDescent, self).__init__(estimator=estimator)

        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        MultiObjectiveProjectedGradientDescent._check_params(self)

        no_preprocessing = self.estimator.preprocessing is None or (
            np.all(self.estimator.preprocessing[0] == 0) and np.all(self.estimator.preprocessing[1] == 1)
        )
        no_defences = not self.estimator.preprocessing_defences and not self.estimator.postprocessing_defences

        self._attack: Union[
            MultiObjectiveProjectedGradientDescentPytorch, ProjectedGradientDescentTensorFlowV2, MultiObjectiveProjectedGradientDescentNumpy
        ]
        if isinstance(self.estimator, PyTorchClassifier) and no_preprocessing and no_defences:
            self._attack = MultiObjectiveProjectedGradientDescentPytorch(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
                strategy = strategy,
                strategy_params = strategy_params,
                strategy_success = strategy_success,
                project=project
            )

        elif isinstance(self.estimator, TensorFlowV2Classifier) and no_preprocessing and no_defences:
            self._attack = ProjectedGradientDescentTensorFlowV2(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
            )

        else:
            self._attack = MultiObjectiveProjectedGradientDescentNumpy(
                estimator=estimator,
                norm=norm,
                eps=eps,
                eps_step=eps_step,
                max_iter=max_iter,
                targeted=targeted,
                num_random_init=num_random_init,
                batch_size=batch_size,
                random_eps=random_eps,
                strategy = strategy,
                strategy_params = strategy_params,
                strategy_success = strategy_success,
                project=project
            )

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        logger.info("Creating adversarial samples.")
        return self._attack.generate(x=x, y=y, **kwargs)

    def set_params(self, **kwargs) -> None:
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

    def _check_params(self) -> None:
        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError("The number of random initialisations has to be of type integer")

        if self.num_random_init < 0:
            raise ValueError("The number of random initialisations `random_init` has to be greater than or equal to 0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if self.norm == np.inf and self.eps_step > self.eps:
            raise ValueError("The iteration step `eps_step` has to be smaller than the total attack `eps`.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")


