import numpy as np
from art.estimators.classification import PyTorchClassifier
from utils.art import MultiOutputClassifier

class MultiOutputPytorchModel(PyTorchClassifier, MultiOutputClassifier):
    def __init__(self, *args, output:list=None, output_name:list=None, **kwargs):
        self._output= output
        self._output_name = output_name
        super(MultiOutputPytorchModel, self).__init__(*args, **kwargs)
        MultiOutputClassifier.__init__(self, *args, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import torch  # lgtm [py/repeated-import]

        # Put the model in the eval mode
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = [np.zeros((x_preprocessed.shape[0], *t), dtype=np.float32) for t in self.nb_classes]
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            with torch.no_grad():
                model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            output = model_outputs[-1]
            results[begin:end] = [output.get(e).detach().cpu().numpy() for e in self._output_name]

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions



class MultiOutputPytorchClassifier(PyTorchClassifier, MultiOutputClassifier):
    def __init__(self, *args, **kwargs):
        super(MultiOutputPytorchClassifier, self).__init__(*args, **kwargs)
        MultiOutputClassifier.__init__(self, *args, **kwargs)

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import torch  # lgtm [py/repeated-import]

        # Put the model in the eval mode
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = np.zeros((x_preprocessed.shape[0], *self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            with torch.no_grad():
                model_outputs = self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            output = model_outputs[-1]
            results[begin:end] = output.detach().cpu().numpy()

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions
