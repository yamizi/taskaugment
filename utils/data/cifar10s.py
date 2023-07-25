"""
Adapted fom : https://github.com/imrahulr/adversarial_robustness_pytorch/blob/main/core/data/cifar10s.py
"""

import torch

import torchvision
import torchvision.transforms as transforms

import re
import os
import numpy as np

from .semisup import SemiSupervisedDataset
from .semisup import SemiSupervisedSampler
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10

def load_cifar10artificial(data_dir, use_augmentation=False, train=True,
                  aux_data_filename='cifar10_ddpm.npz', data_subset=1, concat_original=True):

    test_transform = transforms.Compose([transforms.ToTensor()])

    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor()])
    else:
        train_transform = test_transform



    npzfile = np.load(os.path.join(data_dir, aux_data_filename))
    nb_items = int(len(npzfile['label']) * data_subset)

    images = torch.Tensor(npzfile['image'][:nb_items])
    labels = torch.IntTensor(npzfile['label'][:nb_items])

    if concat_original:
        original = CIFAR10(root='data/', train=train, download=True, transform=None)
        images = torch.cat([images, torch.Tensor(original.data)],0)
        labels = torch.cat([labels, torch.Tensor(original.targets)], 0)

    dataset = TensorDataset(images, labels)
    #dataset = TensorDataset(torch.Tensor(original.data), torch.Tensor(original.targets))

    return dataset


def load_cifar10s(data_dir, use_augmentation=False, aux_take_amount=None, 
                  aux_data_filename='ti_500K_pseudo_labeled.pickle',
                  train=True):
    """
    Returns semisupervised CIFAR10 train, test datasets and dataloaders (with Tiny Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset. 
    """
    data_dir = re.sub('cifar10s', 'cifar10', data_dir)
    test_transform = transforms.Compose([transforms.ToTensor()])
    if use_augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(0.5), 
                                              transforms.ToTensor()])
    else: 
        train_transform = test_transform
    
    dataset = SemiSupervisedCIFAR10(base_dataset='cifar10', root=data_dir, train=train, download=True,
                                          transform=train_transform, aux_data_filename=aux_data_filename, 
                                          add_aux_labels=True, aux_take_amount=aux_take_amount, validation=False)

    return dataset


class SemiSupervisedCIFAR10(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for CIFAR10.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'cifar10', 'Only semi-supervised cifar10 is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.CIFAR10(train=train, **kwargs)
        self.dataset_size = len(self.dataset)


