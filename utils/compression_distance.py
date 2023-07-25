#!/usr/bin/python
# -*- coding: utf-8 -*-

import snappy
import bz2
import zlib
import base64

class NormalizedCompressionDistance(object):

    def __init__(self, compressor):
        self.compressor = compressor

    def ncd(self, a, b):
        compressed_a = self.compressor.compress(a)
        compressed_b = self.compressor.compress(b)
        c = a+b
        return (float(len(self.compressor.compress(c)) - len(min(compressed_a, compressed_b)))) / float(
            len(max(compressed_a,
                    compressed_b)))


if __name__ == '__main__':
    s1 = "the waiter was not eager"
    s2 = 'computer graphic memory'
    s3 = 'hardware computer engineer'
    s4 = 'the service was bad'

    import torch
    img1 = torch.rand(1, 1, 256, 256)
    img2 = torch.ones_like(img1)
    img3 = torch.rand(1, 1, 256, 256)

    n = NormalizedCompressionDistance(snappy)

    print ('lesser, the better =====>')
    print("imgs", n.ncd((img1.numpy()), (img3.numpy())))
    print     (s1, s2,n.ncd(s1, s2))
    print    (s1, s3,n.ncd(s1, s3))
    print    (s1, s4, n.ncd(s1, s4))
    print    (s2, s3, n.ncd(s2, s3))
    print    (s2, s4, n.ncd(s2, s4))
    print    (s3, s4, n.ncd(s3, s4))
