import numpy as np
import itertools

def nb_arrangements(total_classes,nb_classes):
    return int(np.math.factorial(total_classes)/np.math.factorial(total_classes-nb_classes))

def count_occurences(y):
    from collections import Counter

    arg = np.argmax(y, axis=1)
    c = Counter(arg)
    classes = [get_comb_label(e) for e in c.keys()]
    return list(zip(classes, c.values()))


def compute_tsne(hidden_features, max_elements=None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    pca = PCA(n_components=20)
    pca_result = pca.fit_transform(hidden_features)
    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))
    ##Variance PCA: 0.993621154832802

    # Run T-SNE on the PCA features.
    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(pca_result[:max_elements]) if max_elements is not None else tsne.fit_transform(pca_result)

    return tsne_results

def get_comb_label(i, nb_classes=10,nb_combinations = 2):
    return (i//10,i%10)
    comb = list(itertools.combinations(np.arange(nb_classes), nb_combinations))
    t = [(0,0)]+comb + [(j, i) for (i, j) in comb]
    t =comb + [(j, i) for (i, j) in comb]
    t.sort()
    return str(t[i])


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
