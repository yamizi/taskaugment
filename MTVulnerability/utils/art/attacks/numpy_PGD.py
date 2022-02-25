import numpy as np
import tensorflow as tf
from art.utils import (
    get_labels_np_array,
    check_and_transform_label_format,
    random_sphere,
    projection,
)

from art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassifierGradients,
)

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)

from art.config import ART_NUMPY_DTYPE
from typing import Optional
from utils.art import compute_success

import logging
logger = logging.getLogger(__name__)


class MultiObjectiveProjectedGradientDescentNumpy(ProjectedGradientDescentNumpy):
    def __init__(
        self,
        estimator,
        norm=np.inf,
        eps=0.3,
        eps_step=0.1,
        max_iter=100,
        targeted=False,
        num_random_init=0,
        batch_size=32,
        random_eps=False,
        strategy:str = "WEIGHTED_TASKS",
        strategy_params: dict = {},
        strategy_success:str = "MEAN",
        project:bool = True
    ):
        """
        Create a :class:`.ProjectedGradientDescentNumpy` instance.

        :param estimator: An trained estimator.
        :type estimator: :class:`.BaseEstimator`
        :param norm: The norm of the adversarial perturbation supporting np.inf, 1 or 2.
        :type norm: `int`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param eps_step: Attack step size (input variation) at each iteration.
        :type eps_step: `float`
        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature
                           suggests this for FGSM based training to generalize across different epsilons. eps_step
                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with
                           PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        :type random_eps: `bool`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting
                                at the original input.
        :type num_random_init: `int`
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :type batch_size: `int`
        """
        super(MultiObjectiveProjectedGradientDescentNumpy, self).__init__(
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

        self._project = project
        self._cur_iter = 0
        self._strategy = strategy
        self._strategy_params = strategy_params
        self._strategy_success = strategy_success

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.

        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        # Check whether random eps is enabled
        self._random_eps()

        if isinstance(self.estimator, ClassifierMixin):
            # Set up targets
            targets = self._set_targets(x, y)

            # Get the mask
            mask = self._get_mask(x, **kwargs)

            # Start to compute adversarial examples
            adv_x_best = None
            rate_best = None

            for _ in range(max(1, self.num_random_init)):
                adv_x = x.astype(ART_NUMPY_DTYPE)

                for i_max_iter in range(self.max_iter):
                    self.cur_iter = i_max_iter
                    adv_x = self._compute(
                        adv_x,
                        x,
                        targets,
                        mask,
                        self.eps,
                        self.eps_step,
                        self._project,
                        self.num_random_init > 0 and i_max_iter == 0,
                    )

                if self.num_random_init > 1:
                    rate = 100 * compute_success(
                        self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size,
                        strategy=self._strategy_success,                         # type: ignore
                    )
                    if rate_best is None or rate > rate_best or adv_x_best is None:
                        rate_best = rate
                        adv_x_best = adv_x
                else:
                    adv_x_best = adv_x

            logger.info(
                "Success rate of attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,
                    strategy=self._strategy_success,  # type: ignore
                ),
            )
        else:
            if self.num_random_init > 0:
                raise ValueError("Random initialisation is only supported for classification.")

            # Set up targets
            targets = self._set_targets(x, y, classifier_mixin=False)

            # Get the mask
            mask = self._get_mask(x, classifier_mixin=False, **kwargs)

            # Start to compute adversarial examples
            adv_x = x.astype(ART_NUMPY_DTYPE)

            for i_max_iter in range(self.max_iter):
                adv_x = self._compute(
                    adv_x,
                    x,
                    targets,
                    mask,
                    self.eps,
                    self.eps_step,
                    self._project,
                    self.num_random_init > 0 and i_max_iter == 0,
                )

            adv_x_best = adv_x

        return adv_x_best

    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        eps: float,
        eps_step: float,
        project: bool,
        random_init: bool,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE)

            # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = [e[batch_index_1:batch_index_2] for e in y]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels, mask_batch)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, eps_step)

            if project:
                perturbation = projection(
                    x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], eps, self.norm
                )
                x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv

    def _set_targets(self, x, y, classifier_mixin=True):
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :type classifier_mixin: `bool`
        :return: The targets.
        :rtype: `np.ndarray`
        """
        if classifier_mixin:
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                predictions = self.estimator.predict(x, batch_size=self.batch_size)
                targets = [get_labels_np_array(e) for e in predictions]
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets


    def _one_task_gradient(self, gradients, task_index=0):
        return gradients[int(task_index)]


    def _alternate_tasks_gradient(self, gradients):
        nb_gradients = len(gradients)
        task_index = self._cur_iter % nb_gradients
        return gradients[task_index]

    def _weighted_tasks_gradient(self, gradients, weights=None):
        nb_gradients = len(gradients)
        if weights is None:
            weights = np.ones(nb_gradients) / nb_gradients

        gradient = np.sum(np.array([weights[i] * gradients[i] for i in range(nb_gradients)]), axis=0)
        #gradient = tf.reduce_sum(np.array([weights[i]*gradients[i] for i in range(nb_gradients)]), axis=0)
        return gradient

    def _get_gradient_strategy(self, gradients):
        strategies = {
            "ONE_TASK":self._one_task_gradient, "WEIGHTED_TASKS":self._weighted_tasks_gradient, "ALTERNATE_TASKS":self._alternate_tasks_gradient
        }

        strategy = self._strategy.split("/")
        params = (*self._strategy_params, *strategy[1]) if len(strategy) > 1 else self._strategy_params
        gradient = strategies[strategy[0]](gradients, *params)

        return gradient#.eval(session=tf.compat.v1.Session())

    def _compute_perturbation(self, batch: np.ndarray, batch_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradients(batch, y_=batch_labels)
        grad = self._get_gradient_strategy(grad) * (1 - 2 * int(self.targeted))

        # Record gradient
        if self.estimator.record:
            self.estimator.records["gradients"].append(grad)

        # Apply norm bound
        if self.norm == np.inf:
            grad = np.sign(grad)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
        assert batch.shape == grad.shape

        if mask is None:
            return grad
        else:
            return grad * (mask.astype(ART_NUMPY_DTYPE))

