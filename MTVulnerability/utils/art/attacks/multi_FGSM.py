import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierMixin
from art.utils import (
    get_labels_np_array,
    check_and_transform_label_format,
)

from utils.art import compute_success

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

from typing import Optional, Union

from art.attacks.evasion.fast_gradient import FastGradientMethod

import logging
logger = logging.getLogger(__name__)

class MultiObjectiveFastGradientDescent(FastGradientMethod):

    def __init__(self,**kwargs):
        super(MultiObjectiveFastGradientDescent, self).__init__(**kwargs)


    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be
                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially
                     perturbed.
        :type mask: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        if isinstance(self.estimator, ClassifierMixin):
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                predictions = self.estimator.predict(x, batch_size=self.batch_size)
                y = [get_labels_np_array(
                     e # type: ignore
                ) for e in predictions]
            y = y / np.sum(y, axis=1, keepdims=True)

            mask = kwargs.get("mask")
            if mask is not None:
                # ensure the mask is broadcastable:
                if len(mask.shape) > len(x.shape) or mask.shape != x.shape[-len(mask.shape) :]:
                    raise ValueError("mask shape must be broadcastable to input shape")

            # Return adversarial examples computed with minimal perturbation if option is active
            rate_best: Optional[float]
            if self.minimal:
                logger.info("Performing minimal perturbation FGM.")
                adv_x_best = self._minimal_perturbation(x, y, mask)
                rate_best = 100 * compute_success(
                    self.estimator, x, y, adv_x_best, self.targeted, batch_size=self.batch_size,  # type: ignore
                )
            else:
                adv_x_best = None
                rate_best = None

                for _ in range(max(1, self.num_random_init)):
                    adv_x = self._compute(x, x, y, mask, self.eps, self.eps, self._project, self.num_random_init > 0,)

                    if self.num_random_init > 1:
                        rate = 100 * compute_success(
                            self.estimator, x, y, adv_x, self.targeted, batch_size=self.batch_size,  # type: ignore
                        )
                        if rate_best is None or rate > rate_best or adv_x_best is None:
                            rate_best = rate
                            adv_x_best = adv_x
                    else:
                        adv_x_best = adv_x

            logger.info(
                "Success rate of FGM attack: %.2f%%",
                rate_best
                if rate_best is not None
                else 100
                * compute_success(
                    self.estimator,  # type: ignore
                    x,
                    y,
                    adv_x_best,
                    self.targeted,
                    batch_size=self.batch_size,
                ),
            )

        else:
            if self.minimal:
                raise ValueError("Minimal perturbation is only supported for classification.")

            if kwargs.get("mask") is not None:
                raise ValueError("Mask is only supported for classification.")

            if y is None:
                # Throw error if attack is targeted, but no targets are provided
                if self.targeted:
                    raise ValueError("Target labels `y` need to be provided for a targeted attack.")

                # Use model predictions as correct outputs
                logger.info("Using model predictions as correct labels for FGM.")
                y = self.estimator.predict(x, batch_size=self.batch_size)

            adv_x_best = self._compute(x, x, y, None, self.eps, self.eps, self._project, self.num_random_init > 0,)

        return adv_x_best


