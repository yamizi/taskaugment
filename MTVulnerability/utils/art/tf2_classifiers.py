import numpy as np
from art.estimators.classification import TensorFlowV2Classifier


class MultiOutputTF2Classifier(TensorFlowV2Classifier):
    def __init__(self,**kwargs):
        super(MultiOutputTF2Classifier, self).__init__(**kwargs)


        output = kwargs.get("output",None)
        if output is not None:
            self._nb_classes = np.array([e.get_shape()[-1] for e in self._output])

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        results = [np.zeros((x_preprocessed.shape[0], e), dtype=np.float32) for e in self._nb_classes]
        # results = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=np.float32)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Run prediction
            # results[begin:end] = self._model(x_preprocessed[begin:end])

            p = self._model.predict([x_preprocessed[begin:end]])
            for i, e in enumerate(results):
                results[i][begin:end] = p[i]

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)
        return predictions

    def loss_gradient_framework(self, x: "tf.Tensor", y: "tf.Tensor", **kwargs) -> "tf.Tensor":
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Input with shape as expected by the model.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: Gradients of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._loss_object is None:
            raise ValueError("Loss object is necessary for computing the loss gradient.")

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                predictions = self._model(x)

                if self._reduce_labels:
                    loss = self._loss_object(tf.argmax(y, axis=1), predictions)
                else:
                    loss = self._loss_object(y, predictions)

                loss_grads = tape.gradient(loss, x)

        else:
            raise NotImplementedError("Expecting eager execution.")

        return loss_grads

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :param y: Correct labels, one-vs-rest encoding.
        :return: Array of gradients of the same shape as `x`.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if self._loss_object is None:
            raise TypeError(
                "The loss function `loss_object` is required for computing loss gradients, but it has not been "
                "defined."
            )

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y, fit=False)
        x_, y_preprocessed = zip(*[self._apply_preprocessing(x, y_, fit=False) for y_ in y])
        x_preprocessed = x_
        y = np.concatenate(y_preprocessed, axis=0)

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                x_preprocessed_tf = tf.concat([tf.convert_to_tensor(e) for e in x_preprocessed], 0)
                tape.watch(x_preprocessed_tf)
                predictions = self._model(x_preprocessed_tf)
                if self._reduce_labels:
                    loss = self._loss_object(np.argmax(y, axis=1), predictions)
                else:
                    loss = self._loss_object(y, predictions)

            gradients = tape.gradient(loss, x_preprocessed_tf).numpy()

        else:
            raise NotImplementedError("Expecting eager execution.")

        # Apply preprocessing gradients
        gradients = self._apply_preprocessing_gradient(x, gradients)
        return gradients

