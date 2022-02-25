"""
CKA method based on https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
@inproceedings{pmlr-v97-kornblith19a,
  title = {Similarity of Neural Network Representations Revisited},
  author = {Kornblith, Simon and Norouzi,
      Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages = {3519--3529},
  year = {2019},
  volume = {97},
  month = {09--15 Jun},
  publisher = {PMLR}
}
"""

import numpy as np
import os
import sys, time, math

from multiprocessing import Process, Pool
from keras import layers, models
sys.path.append("./")

from utils.cka_alt import linear_CKA

class BatchedCKA(object):

    def __init__(self, batch_size,X1, X2,model, sorted_layers, model2=None):
        super().__init__()

        self.batch_size,self.X1, self.X2,self.model, self.sorted_layers = batch_size,X1, X2,model, sorted_layers
        self.model2 = model2 if model2 else model
        
    def run(self,batch):
        activ = []
        batch_size,X1, X2,model, sorted_layers = self.batch_size,self.X1, self.X2,self.model, self.sorted_layers
        min_ = batch*batch_size
        max_ = min((batch+1)*self.batch_size, len(X1))
        x1 = X1[np.arange(min_, max_)]
        x2 = X2[np.arange(min_, max_)]
        act1, act2 = list(get_activation_layers(
            model, x1, sorted_layers)), list(get_activation_layers(model, x2, sorted_layers))

        for layer, _ in enumerate(act1):
            layer1, layer2 = act1[layer], act2[layer]
            act = []
            if layer1 is None:
                continue
            for i, input1 in enumerate(layer1):
                cka = linear_CKA(input1, layer2[i])#feature_space_linear_cka(input1, layer2[i])
                act.append(cka)
            activ.append(act)

        return np.array(activ)

def get_activation_layers(model, x, sorted_layers=True):
    sys.path.append("./keract/")
    from keract import get_activations

    act = get_activations(model, x)

    if not sorted_layers:
        return act.values()
    layers = dict([(k[0:k.index("/")],v) for (k,v) in act.items() if k[0:5]!="input"])
    ls = [layers.get(y.name) for y in model.layers]

    if ls[1] is None:
        layers = dict([(k[0:k.rfind("_")], v) for (k, v) in act.items() if k[0:5] != "input"])
        ls = [layers.get(y.name) for y in model.layers]

    return ls


def compare_layers(model, X1, X2, run_parallel=0, batch_size=64, sorted_layers=True, model2=None):
    begin = time.time()
    if not hasattr(model, "_is_compiled"):
        # model = model.model
        # model._is_compiled = True

        input_layer = layers.Input(batch_shape=model.layers[0].input_shape)
        prev_layer = input_layer
        for layer in model.layers:
            prev_layer = layer(prev_layer)

        model = models.Model([input_layer], [prev_layer])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model._is_compiled = True

    nb_layers = len(model.layers)
    nb_inputs = len(X1)
    batch_size = nb_inputs if not run_parallel else batch_size
    
    if run_parallel:
        nb_batches = math.ceil(nb_inputs / batch_size)
        batch_size = int(nb_inputs/nb_batches)
        nb_inputs = nb_batches*batch_size
        f = BatchedCKA(batch_size,X1[:nb_inputs], X2[:nb_inputs],model, sorted_layers)
        print("{}x{}:{}".format(nb_batches,batch_size,nb_inputs))
        

        with Pool(nb_batches) as p:
            activations = np.array(p.map(f.run, range(nb_batches)))
            activations = np.swapaxes(activations,0,1)
            activations = np.reshape(activations, (nb_layers,nb_inputs))
    else:
        f = BatchedCKA(batch_size,X1, X2,model, sorted_layers, model2=model2)
        activations = f.run(0)

    print("activation {}, time {}, workers {}".format(activations.shape, int(time.time()-begin), run_parallel))
    return activations


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)


def test_cka():

    np.random.seed(1337)
    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10) + X

    cka_from_examples = cka(gram_linear(X), gram_linear(Y))
    cka_from_features = feature_space_linear_cka(X, Y)

    print('Linear CKA from Examples: {:.5f}'.format(cka_from_examples))
    print('Linear CKA from Features: {:.5f}'.format(cka_from_features))
    np.testing.assert_almost_equal(cka_from_examples, cka_from_features)
