import numpy as np
from typing import Optional, Union

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)

import logging
logger = logging.getLogger(__name__)

class MultiObjectiveProjectedGradientDescentPytorch(ProjectedGradientDescentPyTorch):
    def __init__(
        self,
        estimator: Union["PyTorchClassifier"],
        norm: Union[int, float, str] = np.inf,
        eps: Union[int, float, np.ndarray] = 0.3,
        eps_step: Union[int, float, np.ndarray] = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 32,
        random_eps: bool = False,
        verbose: bool = True,
        strategy:str = "WEIGHTED_TASKS",
        strategy_params: dict = {},
        strategy_success:str = "MEAN",
        project:bool = True,
    ):
        """
        Create a :class:`.ProjectedGradientDescentPyTorch` instance.
        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step is
                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD
                           is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :param max_iter: The maximum number of iterations.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """

        super(MultiObjectiveProjectedGradientDescentPytorch, self).__init__(
            estimator=estimator,
            norm=norm,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            num_random_init=num_random_init,
            batch_size=batch_size,
            random_eps=random_eps,
            verbose=verbose,
        )

        self._project = project
        self._cur_iter = 0
        self._strategy = strategy
        self._strategy_params = strategy_params
        self._strategy_success = strategy_success
