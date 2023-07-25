"""https://github.com/eccv22-ood-workshop/ROBIN-dataset"""
import os
import pandas as pd
import torch
import numpy as np
from torchvision import transforms
from skimage.io import imread
from torchxrayvision.datasets import Dataset
from torch.nn.functional import one_hot

from utils.multitask_models import get_tasks_to_params

class ROBIN_Dataset(Dataset):

    def __init__(self,
                 data_folder,
                 csv_path="utils/data",
                 train = True,
                 transform=None,
                 data_aug=None,
                 data_subset = 1,
                 seed=0,
                 ):

        super(ROBIN_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.is_binary = False

        self.transform = transforms.Compose([transforms.ToTensor()]) if transform else None
        self.data_aug = transforms.Compose([ transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()]) if data_aug else None

        self.pathologies = ['multilabel','shape', 'pose', 'texture', 'context', 'weather']
        params = get_tasks_to_params([10])
        self.nb_classes = {k: params["class_object#{}".format(k)].get('num_classes') for k in self.pathologies}

        self.imgpath = "{}-cls-{}".format(data_folder,"train" if train else "val")
        self.csvpath = "{}/robin_{}.csv".format(csv_path,"train" if train else "val")
        self.csv = pd.read_csv(self.csvpath, delimiter=";").fillna(0)

        if data_subset < 1:
            self.csv = self.csv.head(int(data_subset*len(self.csv)))

        self.classes = ["car","aeroplane","bicycle","boat","bus","chair","diningtable","motorbike","sofa","train"]
        self.csv["class"] = self.csv["class"].replace({k:i for (i,k) in enumerate(self.classes)})

        self.labels = self.csv[["class","shape","pose","texture","context","weather"]].values


    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        labels = self.labels[idx]
        robin_labels = one_hot(torch.Tensor([labels[0]]).long(), self.nb_classes['multilabel'])
        sample["lab"] = robin_labels

        lbls = {lb:one_hot(torch.Tensor([labels[i]]).long(), self.nb_classes[lb]) for (i,lb) in enumerate(self.pathologies) if lb!='multilabel'}
        sample = {**sample, **lbls}

        img_class = self.classes[int(labels[0])]
        img_path = os.path.join(self.imgpath, img_class,self.csv.at[idx, "path"])
        img = imread(img_path)

        sample["img"] = img

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return self.enrich(sample, idx)
