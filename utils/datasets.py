import os
import torch
import numpy as np

from torch.utils.data import DataLoader, SubsetRandomSampler

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, clean_dataset, return_label=False):
        self.base = clean_dataset
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]
        if self.return_label:
            return image_orig,  label
        else:
            return image_orig


def build_art_dataset(batch_size,dataset="cifar10_train", dataset_size=32, nb_batches=None, artificial_task="hog", dataset_params={}):
    from utils.loader import HOGTensorDataset, SIFTTensorDataset, GaborTensorDataset
    if artificial_task=="hog":
        train_dataset = HOGTensorDataset(tensors=None, dataset=dataset, dataset_size=dataset_size, dataset_params=dataset_params)
    if artificial_task=="sift":
        train_dataset = SIFTTensorDataset(tensors=None, dataset=dataset, dataset_size=dataset_size, nb_keypoints=15)
    elif artificial_task=="gabor":
        train_dataset = GaborTensorDataset(tensors=None, dataset=dataset, dataset_size=dataset_size, dataset_params=dataset_params)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    X = []
    X_original = []
    Y = []
    for batch_idx, (x_train, y_train, x_original) in enumerate(train_loader):
        X.append(x_train)
        Y.append(y_train)
        X_original.append(x_original)

        if nb_batches is not None and batch_idx>=nb_batches:
            break

    return torch.cat(X,0),torch.cat(Y,0),torch.cat(X_original,0)

def split_dataset(dataset,split, batch_size,dataset_size=None, shuffle=False):
    if dataset_size is None:
        dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    val_split_index = int(np.floor(split * dataset_size))
    train_idx, val_idx = dataset_indices[val_split_index:], dataset_indices[:val_split_index]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, sampler=val_sampler)

    return (train_loader, val_loader)

def get_multitask_loader(X,Y, *args, parameters=None, batch_size=32, split=False):
    from utils.loader import MultiTaskDataset
    tasks = ["x"] + parameters.get("tasks")
    data = zip(*[X, Y, *args])
    dataset = MultiTaskDataset([dict(zip(tasks, l)) for l in data], params=parameters, length=len(X))

    if split:
        loader = split_dataset(dataset,split, batch_size)

    else:
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)

    return loader

def get_dataset(parameters, shuffle=False, return_loader=False, split=False, transform=None):

    train = parameters.get("train",True)
    dataset = parameters.get("dataset","")
    batch_size = parameters.get("batch_size",16)

    data_folder = parameters.get("data_folder", "~/datasets")
    loader = None
    img_dataset = None

    if dataset.find("hog")>-1 or dataset.find("sift")>-1 or dataset.find("gabor")>-1:
        d = dataset.split("_") #dataset AUX_MAIN_TRAIN/TEST_IMGSIZE_[PARAM_VAL]^N
        artificial_task = d[0]
        if os.path.exists(os.path.join("data",dataset+".pth")):
            Art_X = torch.load(os.path.join("data",dataset+".pth"))
            if dataset.find("gabor") >-1:
                Art_X = Art_X.reshape((Art_X.shape[0], -1))
            Y = torch.load(os.path.join("data", dataset+"_labels.pth"))
            X = torch.load(os.path.join("data", dataset+"_original.pth")).squeeze(1)
        else:
            dataset_size = int(d[3]) if len(d)>3 else 32
            dataset_params = d[4:] if len(d)>4 else []
            dataset_params = dict([dataset_params[i:i + 2] for i in range(0, len(dataset_params), 2)])
            nb_batches = parameters.get("nb_batches", 16)

            Art_X,Y,X = build_art_dataset(batch_size, dataset="_".join(d[1:3]), dataset_size=dataset_size,nb_batches=nb_batches, artificial_task=artificial_task, dataset_params=dataset_params)
            torch.save(Art_X,os.path.join("data",dataset+".pth"))
            torch.save(Y, os.path.join("data", dataset + "_labels.pth"))
            torch.save(X, os.path.join("data", dataset + "_original.pth"))


        loader = get_multitask_loader(X.numpy(),Y.numpy(), Art_X.numpy(), parameters=parameters, batch_size=batch_size, split=split)

    elif dataset.find("div2k")>-1:
        from utils.loader import DataLoader as PathDataLoader
        loader = PathDataLoader('../SteganoGAN/research/data/div2k/val/', limit=np.inf, shuffle=shuffle,
                            batch_size=parameters.get("batch_size"), transform=transform)

    elif dataset.find("imagenette") > -1:
        from utils.loader import DataLoader as PathDataLoader
        loader = PathDataLoader('../datasets/imagenette2/val', limit=np.inf, shuffle=shuffle,
                                batch_size=parameters.get("batch_size"), transform=transform)

    else:

        if dataset.find("robin") > -1:
            from utils.domain_datasets.robin_multitask_datasets import ROBIN_Dataset
            data_folder = os.path.join(parameters.get("dataset_dir", "~/datasets"), "ROBIN")
            img_dataset = ROBIN_Dataset(train=train, transform=True, data_aug=parameters.get("data_aug"),
                                          data_subset=parameters.get("data_subset"),
                                          seed=parameters.get("seed"),
                                          data_folder=data_folder)

        elif dataset.find("imagenetR") > -1:
            from utils.domain_datasets.imagenetR_multitask_datasets import ImageNetR_Dataset
            data_folder = os.path.join(parameters.get("dataset_dir", "~/datasets"), "imagenet-r")
            img_dataset = ImageNetR_Dataset(train=train, transform=True, data_aug=parameters.get("data_aug"),
                                           data_subset=parameters.get("data_subset"),
                                           seed=parameters.get("seed"),data_folder=data_folder)

        elif dataset.find("cifar100") > -1:
            from utils.domain_datasets.cifar10_multitask_datasets import CIFAR100_Dataset
            img_dataset = CIFAR100_Dataset(train=train, transform=True, data_aug=parameters.get("data_aug"),
                                           data_subset=parameters.get("data_subset"),
                                           seed=parameters.get("seed"))

        elif dataset.find("cifar10_100") > -1:
            from utils.domain_datasets.cifar10_multitask_datasets import CIFAR10_100_Dataset
            img_dataset = CIFAR10_100_Dataset(train=train, transform=True, data_aug=parameters.get("data_aug"),
                                              data_subset=parameters.get("data_subset"),
                                              seed=parameters.get("seed"))

        elif dataset.find("cifar10") > -1:
            from utils.domain_datasets.cifar10_multitask_datasets import CIFAR10_Dataset
            data_folder = os.path.join(parameters.get("dataset_dir", "~/datasets"), "CIFAR10")
            img_dataset = CIFAR10_Dataset(train=train, transform=True, data_aug=parameters.get("data_aug"),
                                          data_subset=parameters.get("data_subset"),
                                          seed=parameters.get("seed"), augment=parameters.get("augment", ""),
                                          data_folder=data_folder)

        elif dataset.find("imageNet") > -1:
            import torchvision.datasets as datasets

            root_folder = "{}/imagenet/ILSVRC2012/train/".format(data_folder) if dataset.find("train") > -1 \
                else "{}/imagenet/ILSVRC2012/val/".format(data_folder)
            img_dataset = datasets.ImageFolder(root_folder, transform=transform)

        loader = DataLoader(img_dataset, shuffle=shuffle,
                                batch_size=parameters.get("batch_size"))


    if return_loader and loader is not None:
        return loader

    return img_dataset
