import sys, os
sys.path.append(".")

import numpy as np
import torch
from comet_ml import Experiment
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
from art.data_generators import DataGenerator
from typing import Union

from utils.models import get_model_dataset

class XrayDataGenerator(DataGenerator):
    """
    Wrapper class on top of the PyTorch native data loader :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, iterator: "torch.utils.data.DataLoader", size: int, batch_size: int) -> None:
        """
        Create a data generator wrapper on top of a PyTorch :class:`DataLoader`.

        :param iterator: A PyTorch data generator.
        :param size: Total size of the dataset.
        :param batch_size: Size of the minibatches.
        """
        from torch.utils.data import DataLoader

        super().__init__(size=size, batch_size=batch_size)
        if not isinstance(iterator, DataLoader):
            raise TypeError(f"Expected instance of PyTorch `DataLoader, received {type(iterator)} instead.`")

        self._iterator: DataLoader = iterator
        self._current = iter(self.iterator)

    def get_batch(self) -> tuple:
        """
        Provide the next batch for training in the form of a tuple `(x, y)`. The generator should loop over the data
        indefinitely.

        :return: A tuple containing a batch of data `(x, y)`.
        :rtype: `tuple`
        """
        try:
            batch = next(self._current)
        except StopIteration:
            self._current = iter(self.iterator)
            batch = next(self._current)

        #batch = [batch.get("img").float(), batch.get("lab").nan_to_num(0)]
        batch = [batch.get("img").float(),batch.get("lab")]

        for i, item in enumerate(batch):
            batch[i] = item.data.cpu().numpy()

        return tuple(batch)

def loss_gradient(  # pylint: disable=W0221
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Union[np.ndarray, "torch.Tensor"],
        training_mode: bool = False,
        **kwargs,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  `(nb_samples,)`.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
                              Note on RNN-like models: Backpropagation through RNN modules in eval mode raises
                              RuntimeError due to cudnn issues and require training mode, i.e. RuntimeError: cudnn RNN
                              backward can only be called in training mode. Therefore, if the model is an RNN type we
                              always use training mode but freeze batch-norm and dropout layers if
                              `training_mode=False.`
        :return: Array of gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.train(mode=training_mode)

        # Backpropagation through RNN modules in eval mode raises RuntimeError due to cudnn issues and require training
        # mode, i.e. RuntimeError: cudnn RNN backward can only be called in training mode. Therefore, if the model is
        # an RNN type we always use training mode but freeze batch-norm and dropout layers if training_mode=False.
        if self.is_rnn:
            self._model.train(mode=True)
            if not training_mode:
                self.set_batchnorm(train=False)
                self.set_dropout(train=False)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                x_grad = x.clone().detach().requires_grad_(True)
            else:
                x_grad = torch.tensor(x).to(self._device)
                x_grad.requires_grad = True
            if isinstance(y, torch.Tensor):
                y_grad = y.clone().detach()
            else:
                y_grad = torch.tensor(y).to(self._device)
            inputs_t, y_preprocessed = self._apply_preprocessing(x_grad, y=y_grad, fit=False, no_grad=False)
        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)
            x_grad = torch.from_numpy(x_preprocessed).to(self._device)
            x_grad.requires_grad = True
            inputs_t = x_grad
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        # Compute the gradient and return
        model_outputs = self._model(inputs_t)
        outputs = model_outputs[-1]
        targets = labels_t
        loss = torch.zeros(1).to(self._device).float()
        for task in range(targets.shape[1]):
            task_output = outputs[:, task]
            task_target = targets[:, task]
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                task_loss = self._loss(task_output.float(), task_target.float())
                loss += task_loss


        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward()

        if isinstance(x, torch.Tensor):
            grads = x_grad.grad
        else:
            grads = x_grad.grad.cpu().numpy().copy()  # type: ignore

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

#Monkey patching with xray vision loss estimation
PyTorchClassifier.loss_gradient = loss_gradient

def run(parameters, model_labels = None, experiment=None ):
    print("running", parameters)

    all_models = ["densenet121-res224-all", "resnet50","densenet121-res224-nih", "densenet121-res224-chex",
                     "densenet121-res224-pc", "densenet121-res224-nb", "densenet121-res224-ch",
                     "densenet121-res224-rsna"]

    models_labels = all_models if model_labels is None else model_labels

    for nm in models_labels:
        parameters["model_name"] = nm
        loader, m = get_model_dataset(parameters, device=parameters.get("device","cuda"), return_loader=True,
                                      normalized=False, split=False)

        optim = torch.optim.Adam(m.parameters(), lr=parameters.get("lr"), weight_decay=1e-5, amsgrad=True)
        estimator = PyTorchClassifier(m, torch.nn.BCEWithLogitsLoss(), (1,224,224), 18, optimizer = optim)
        attack = ProjectedGradientDescentPyTorch(
            estimator,
            eps=parameters.get("max_eps"),
            eps_step=parameters.get("max_eps"),
            max_iter=parameters.get("max_iter"),
            num_random_init=parameters.get("num_random_init"),
        )
        trainer = AdversarialTrainer(estimator, attack)
        trainer.fit_generator(XrayDataGenerator(loader, size=len(loader), batch_size=parameters.get("batch_size"))
                              ,nb_epochs = parameters.get("num_epochs"))

        if parameters.get("output_dir"):
            model_name = "{}_{}".format(nm,parameters.get("num_epochs"))
            estimator.save(model_name,parameters.get("output_dir"))


if __name__ == '__main__':

    parameters = {"criterion": "bce", "algorithm": "pgd", "max_eps": 4 / 255, "workspace": "aaai22_health",
                  "norm": "Linf", "max_iter": 100, "eps_step": 1/255, "num_random_init": 1, "batch_size": 16,
                  "nb_batches": 4, "lib": "torchattack", "model": "xrayvision", "dataset": "NIH", "reduction": "none",
                  "data_folder": os.path.join("data", 'NIH Chest X-rays'), "lr":0.001}

    models_labels = ["densenet121-res224-all", "resnet50", "densenet121-res224-nih", "densenet121-res224-chex"]

    datasets = [{"dataset": "PC",
                 "data_folder": {"csv": "D://datasets//PC//PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                                 "img": "D://datasets//PC//images-224//images-224"}},
                {"dataset": "NIH", "data_folder": os.path.join("data", 'NIH Chest X-rays')},
                {"dataset": "CHEX",
                 "data_folder": {"csv": "D://datasets//CheXpert//train.csv", "img": "D://datasets//CheXpert"}}]

    datasets = [{"dataset": "CHEX",
                 "data_folder": {"csv": "D://datasets//CheXpert//train.csv", "img": "D://datasets//CheXpert"}}]

    for dt in datasets:
        parameters = {**parameters, **dt}
        run(parameters, name="xray_baseline_robust_training", model_labels=models_labels)
