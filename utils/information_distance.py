# Adopted from: https://github.com/weilu/coop-learning and https://gist.github.com/jwcarr/626cbc80e0006b526688

from math import log

cache = {}

def calculate_mutual_information_and_entropies(X, Y):
    # convert to tuples which are hashable for caching
    X_tuple = (tuple(x) for x in X)
    Y_tuple = (tuple(y) for y in X)

    if (X_tuple, Y_tuple) in cache:
        return cache[X_tuple, Y_tuple]

    n = float(sum([len(x) for x in X]))
    entropy = 0.0
    entropy_x = 0.0
    entropy_y = 0.0
    mi = 0.0

    for i, x in enumerate(X):
        p = len(x) / n
        entropy_x += -(p * log(p, 2))
        for y in Y:
            q = len(y) / n
            if i == 0:
                entropy_y += -(q * log(q, 2))
            r = len(set(x) & set(y)) / n
            if r > 0:
                mi_term = r * log(r / (p * q), 2)
                mi += mi_term
                entropy += mi_term - r * (log(r / p, 2) + log(r / q, 2))

    cache[(X_tuple, Y_tuple)] = (mi, entropy, entropy_x, entropy_y)
    return cache[(X_tuple, Y_tuple)]


def variation_of_information(X, Y):
    mi, entropy, _, _ = calculate_mutual_information_and_entropies(X, Y)
    return entropy - mi


# as defined in http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
def normalized_variation_of_information(X, Y):
    mi, entropy, _, _ = calculate_mutual_information_and_entropies(X, Y)
    return 1 - mi / entropy


def normalized_information_distance(X, Y):
    mi, _, entropy_x, entropy_y = calculate_mutual_information_and_entropies(X, Y)
    return 1 - mi / max(entropy_x, entropy_y)


if __name__ == '__main__':
    s1 = "the waiter was not eager"
    s2 = 'computer graphic memory'
    s3 = 'hardware computer engineer'
    s4 = 'the service was bad'

    import torch
    img1 = torch.rand(1, 1, 256, 256)
    img2 = torch.ones_like(img1)
    img3 = torch.rand(1, 1, 256, 256)

    print ('lesser, the better =====>')
    print("imgs", normalized_information_distance((img1.numpy()), (img3.numpy())))
    print     (s1, s2,normalized_information_distance(s1, s2))
    print    (s1, s3,normalized_information_distance(s1, s3))
    print    (s1, s4, normalized_information_distance(s1, s4))
    print    (s2, s3, normalized_information_distance(s2, s3))
    print    (s2, s4, normalized_information_distance(s2, s4))
    print    (s3, s4, normalized_information_distance(s3, s4))
