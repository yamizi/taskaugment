import os
import torch
import numpy as np
from torchxrayvision.datasets import Dataset
from torchvision import transforms

from torchvision.datasets import CIFAR10, CIFAR100, VisionDataset, STL10
from torch.nn.functional import one_hot

from utils.data.cifar10s import load_cifar10s,load_cifar10artificial

class CIFAR10_100(VisionDataset):

    def one_hot(self, a, num_classes):
        a = np.array(a)
        return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

    def __init__(self, cifar10, cifar100, root, transform, target_transform=None, nb_classes=(10,100), data_subset=1):
        super(CIFAR10_100, self).__init__(root, transform=transform, target_transform=target_transform)

        if data_subset < 1:
            nb_items10 = int(len(cifar10.data) * data_subset)
            nb_items100 = int(len(cifar100.data) * data_subset)

            cifar10.data = cifar10.data[:nb_items10]
            cifar10.targets = cifar10.targets[:nb_items10]

            cifar100.data = cifar10.data[:nb_items100]
            cifar100.targets = cifar10.targets[:nb_items100]

        self.data = np.concatenate([cifar10.data, cifar100.data])
        cifar10_targets = self.one_hot(cifar10.targets, nb_classes[0])
        cifar100_targets = self.one_hot(cifar100.targets, nb_classes[1])
        self.labels = [np.concatenate([cifar10.targets, np.full_like(cifar100.targets, np.nan)]),
                       np.concatenate([np.full_like(cifar10.targets, np.nan), cifar100.targets])]

        self.targets = [
            np.concatenate([cifar10_targets, np.full((cifar100_targets.shape[0], cifar10_targets.shape[1]), np.nan)]),
            np.concatenate([np.full((cifar10_targets.shape[0], cifar100_targets.shape[1]), np.nan), cifar100_targets])]

    def __len__(self):
        return len(self.data)

class CIFAR10_100_Dataset(Dataset):

    def __init__(self,
                 train = True,
                 transform=None,
                 data_aug=None,
                 data_subset = 1,
                 seed=0,
                 resized = 40,
                 cropped=32):

        super(CIFAR10_100_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        root = 'data/'
        self.nb_classes = {'class_object#multilabel': 10, 'class_object#macro': 100}
        self.is_binary = False
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform else None
        self.data_aug = transforms.Compose([transforms.Resize((resized, resized)),
                    transforms.RandomCrop(cropped),
                    transforms.RandomHorizontalFlip()]) if data_aug else None

        self.img_dataset10 = CIFAR10(root=root, train=train, download=True, transform=None)
        self.img_dataset100 = CIFAR100(root=root, train=train, download=True, transform=None)

        self.img_dataset = CIFAR10_100(self.img_dataset10,self.img_dataset100, root=root, transform=None,
                                       data_subset =data_subset, nb_classes=list(self.nb_classes.values()))

        self.pathologies = ["multilabel","macro"]


        # Get our classes.
        labels = np.asarray(self.img_dataset.labels)
        labels = np.expand_dims(labels,1) if len(labels.shape)==1 else labels.T

        self.labels = labels

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)
    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        labels = self.img_dataset.targets
        labels_cifar10 =  torch.Tensor(labels[0][idx]).long()
        labels_cifar100 = torch.Tensor(labels[1][idx]).long()

        sample["lab"] = labels_cifar10 if torch.all(labels_cifar10>-1) else torch.nan
        sample["macro"] = labels_cifar100 if torch.all(labels_cifar100>-1) else torch.nan
        sample["img"] = self.img_dataset.data[idx]

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return self.enrich(sample, idx)

class CIFAR100_Dataset(Dataset):

    def __init__(self,
                 train = True,
                 transform=None,
                 data_aug=None,
                 data_subset = 1,
                 seed=0,
                 resized = 40,
                 cropped=32,
                 ):

        super(CIFAR100_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.is_binary = False

        self.transform = transforms.Compose([transforms.ToTensor()]) if transform else None
        self.data_aug = transforms.Compose([transforms.Resize((resized, resized)),
                    transforms.RandomCrop(cropped),
                    transforms.RandomHorizontalFlip()]) if data_aug else None

        self.img_dataset = CIFAR100(root='data/', train=train, download=True, transform=None)
        if data_subset < 1:
            nb_items = int(len(self.img_dataset) * data_subset)
            self.img_dataset.data = self.img_dataset.data[:nb_items]
            self.img_dataset.targets = self.img_dataset.targets[:nb_items]
        self.pathologies = ["multilabel","macro"]

        coarse_labels = (4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13)

        self.micro_macro = {i:coarse_labels[i] for i in range(len(coarse_labels))}
        self.nb_classes = {'class_object#multilabel':len(coarse_labels), 'class_object#macro':20}

        # Get our classes.
        labels = np.asarray(self.img_dataset.targets)
        macro_labels = np.asarray([self.micro_macro[t] for t in self.img_dataset.targets])

        labels = np.expand_dims(labels, 1) if len(labels.shape) == 1 else labels.T
        macro_labels = np.expand_dims(macro_labels, 1) if len(macro_labels.shape) == 1 else macro_labels.T

        self.labels = np.concatenate([labels, macro_labels], 1)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)
    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx

        labels = self.labels[idx]
        cifar100_labels = one_hot(torch.Tensor([labels[0]]).long(), self.nb_classes['class_object#multilabel'])
        macro_labels = one_hot(torch.Tensor([labels[1]]).long(), self.nb_classes['class_object#macro'])

        sample["lab"] = cifar100_labels
        sample["macro"] = macro_labels
        sample["img"] = self.img_dataset.data[idx]

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return self.enrich(sample, idx)


class CIFAR10_Dataset(Dataset):

    def __init__(self,
                 train = True,
                 transform=None,
                 data_aug=None,
                 data_subset = 1,
                 seed=0,
                 resized = 40,
                 cropped=32,
                 augment="",
                 data_folder = ""
                 ):

        super(CIFAR10_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.is_binary = False
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform else None
        self.data_aug = transforms.Compose([transforms.Resize((resized, resized)),
                    transforms.RandomCrop(cropped),
                    transforms.RandomHorizontalFlip()]) if data_aug else None

        if "ss" in augment:
            aux_data_filename = os.path.join(data_folder, 'ti_500K_pseudo_labeled.pickle')
            print("loading semi-supervised from {}".format(aux_data_filename))
            self.img_dataset = load_cifar10s(data_dir = data_folder, train=train,
                                             aux_data_filename=aux_data_filename)

        elif "gowal" in augment:
            print("loading GAN data")
            self.img_dataset = load_cifar10artificial(data_dir = data_folder, train=train, data_subset=data_subset)
            self.img_dataset.targets = self.img_dataset.tensors[1]
            self.img_dataset.data = self.img_dataset.tensors[0].numpy()

        elif "stl" in augment:
            self.img_dataset = STL10(root='data/', split="train" if train else "test", download=True, transform=None)
            self.img_dataset.targets = self.img_dataset.labels
            self.img_dataset.data = np.moveaxis(self.img_dataset.data,1,3)

        else:
            self.img_dataset = CIFAR10(root='data/', train=train, download=True, transform=None)

        if data_subset < 1 and not "gowal" in augment:

            nb_items = int(len(self.img_dataset) * data_subset)
            self.img_dataset.data = self.img_dataset.data[:nb_items]
            self.img_dataset.targets = self.img_dataset.targets[:nb_items]

            if hasattr(self.img_dataset,"tensors"):
                self.img_dataset.tensors = (self.img_dataset.tensors[0][:nb_items], self.img_dataset.tensors[1][:nb_items])


        self.pathologies = ["multilabel","macro", "detector"]

        #0: vehicle, 1: animal
        self.micro_macro = {0:0,1:0,2:1,3:1,4:1,5:1,6:1,7:1,8:0,9:0}
        self.nb_classes = {'class_object#multilabel':10, 'class_object#macro':2, 'class_object#detector':2}

        # Get our classes.
        labels = np.asarray(self.img_dataset.targets)
        macro_labels = np.asarray([self.micro_macro[t] for t in labels])

        labels = np.expand_dims(labels,1) if len(labels.shape)==1 else labels.T
        macro_labels = np.expand_dims(macro_labels, 1) if len(macro_labels.shape) == 1 else macro_labels.T

        self.labels = np.concatenate([labels,macro_labels],1)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views,
                                                                                       self.data_aug)
    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        labels = self.labels[idx]
        cifar10_labels = one_hot(torch.Tensor([labels[0]]).long(), self.nb_classes['class_object#multilabel'])
        macro_labels = one_hot(torch.Tensor([labels[1]]).long(), self.nb_classes['class_object#macro'])
        sample["lab"] = cifar10_labels
        sample["macro"] = macro_labels
        sample["detector"] =  one_hot(torch.Tensor([np.random.rand()<0.5]).long(), self.nb_classes['class_object#detector'])
        sample["img"] = self.img_dataset.data[idx]

        if self.transform is not None:
            sample["img"] = self.transform(sample["img"])

        if self.data_aug is not None:
            sample["img"] = self.data_aug(sample["img"])

        return self.enrich(sample, idx)
