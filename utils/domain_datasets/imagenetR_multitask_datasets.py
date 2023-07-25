import pandas as pd
import torch
import os
import numpy as np
from torchvision import transforms
from skimage.io import imread
from torchxrayvision.datasets import Dataset
from torch.nn.functional import one_hot
from torchvision.datasets import ImageFolder

from utils.multitask_models import get_tasks_to_params

class ImageNetR_Dataset(Dataset):

    def __init__(self,
                 data_folder,
                 train = True,
                 transform=None,
                 data_aug=None,
                 data_subset = 1,
                 seed=0,
                 ):

        super(ImageNetR_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.is_binary = False

        self.transform = transforms.Compose([transforms.ToTensor()]) if transform else None
        self.data_aug = transforms.Compose([ transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()]) if data_aug else None

        self.folderLoader = ImageFolder(data_folder)
        classes = [img[0].split(os.sep)[-1].split("_")[0] for img in self.folderLoader.imgs]
        self.domains = dict(zip(set(classes),range(15)))
        domains = [self.domains[k] for k in classes]
        self.pathologies = ['multilabel','imagenetr']
        params = get_tasks_to_params([len(self.folderLoader.classes)])
        self.nb_classes = {k:params["class_object#{}".format(k)].get('num_classes') for k in self.pathologies}

        self.classes = self.folderLoader.classes

        self.labels = np.array([self.folderLoader.targets,domains]).T


    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)
    def __len__(self):
        return len(self.folderLoader)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        labels = self.labels[idx]
        imagenetR_labels = one_hot(torch.Tensor([labels[0]]).long(), self.nb_classes['multilabel'])
        sample["lab"] = imagenetR_labels

        lbls = {lb:one_hot(torch.Tensor([labels[i]]).long(), self.nb_classes[lb]) for (i,lb) in enumerate(self.pathologies) if lb!='multilabel'}
        sample = {**sample, **lbls}

        img = self.folderLoader.__getitem__(idx)

        sample["img"] = img[0]

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return self.enrich(sample, idx)
