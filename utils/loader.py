# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage.feature import hog
import cv2 as cv
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from torch.utils.data import Dataset
from robustbench.data import _load_dataset

_DEFAULT_MU = [.5, .5, .5]
_DEFAULT_SIGMA = [.5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(360, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):

        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )


class MultiTaskDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, params=None, normalize=True, torchvision_transform= None, length=None, resized= 256,
                 cropped = 224):

        #tensors = [torch.from_numpy(tensor) if isinstance(tensor,np.ndarray) else tensor for tensor in tensors]


        if torchvision_transform is None:
            transforms_list = [
                transforms.ToTensor()]

            if params.get("data_aug"):
                transforms_list = [
                    transforms.ToTensor(),
                    transforms.Resize((resized, resized)),
                    transforms.RandomCrop(cropped),
                    transforms.RandomHorizontalFlip(),
                ]
            torchvision_transform = transforms.Compose(transforms_list)

        self.transform = torchvision_transform

        tasks = params.get("tasks")
        if tasks is None:
            tasks = list(range(len(tensors)-1))

        self.params = params
        self.tasks = tasks
        self.output_size = params.get("output_size")
        self.length = length

        self.tensors = tensors

        print(len(tensors))

    def __getitem__(self, index):

        item = self.tensors[index]
        x = item["x"].transpose(1,2,0) if item["x"].shape[0]!=item["x"].shape[1] else item["x"]

        state = torch.get_rng_state()
        x = self.transform(x)
        y = {}
        for i, type in enumerate(self.tasks):
            if "class" in type:
                y[type] =torch.nn.functional.one_hot(torch.from_numpy(np.array(item[type])).long(), self.output_size)
            elif type in ["depth_zbuffer", "edge_texture", "keypoints2d", "normal", "reshading", "keypoints3d",
                        "depth_euclidean", "edge_occlusion", "principle_curvature", "hog","autoencoder",
                          "age", "hog", "ae", "rotation", "jigsaw"]:

                torch.set_rng_state(state)
                tsk = item[type].transpose(1,2,0) if item[type].shape[0]!=item[type].shape[1] else item[type]
                y[type] = self.transform(tsk)
            else:
                y[type] = torch.from_numpy(item[type])

        return x,y

    def __len__(self):
        return self.length if self.length is not None else len(self.tensors)


class HOGTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors,orientations=9, pixels_per_cell=4, cells_per_block=2,
                 dataset=None, data_dir: str = './data', dataset_size=32, dataset_params=None):

        if dataset is not None:
            if dataset=="cifar10_train":
                transform_chain = transforms.Compose([transforms.ToTensor(), transforms.Resize((dataset_size,dataset_size))])
                dataset = datasets.CIFAR10(root=data_dir,
                                           train=True,
                                           transform=transform_chain,
                                           download=True)

                tensors = _load_dataset(dataset)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.orientations = orientations if dataset_params is None else dataset_params.get("orientations",orientations)
        self.pixels_per_cell = pixels_per_cell if dataset_params is None else dataset_params.get("pixelspercell",pixels_per_cell)
        self.cells_per_block = cells_per_block if dataset_params is None else dataset_params.get("cellsperblock",cells_per_block)

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        img = x.permute(1,2,0)
        resized_img = img.cpu().numpy()

        hogs = []
        for i in range(resized_img.shape[2]):
            _, hg = hog(resized_img[:,:,i], orientations=self.orientations, pixels_per_cell=(self.pixels_per_cell,self.pixels_per_cell),
                            cells_per_block=(self.cells_per_block,self.cells_per_block), visualize=True)
            hogs.append([hg])

        hog_image = np.concatenate(hogs).transpose(1,2,0)*2
        #hog_image = resize(hog_image, x[0].shape)
        hog_x = torch.from_numpy(hog_image*5).float()

        return hog_x.permute(2,0,1), y, x

    def __len__(self):
        return self.tensors[0].size(0)

class GaborTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, dataset=None, data_dir: str = './data', dataset_size=32, dataset_params=None):

        if dataset is not None:
            if dataset=="cifar10_train":
                transform_chain = transforms.Compose([transforms.ToTensor(), transforms.Resize((dataset_size,dataset_size))])
                dataset = datasets.CIFAR10(root=data_dir,
                                           train=True,
                                           transform=transform_chain,
                                           download=True)

                tensors = _load_dataset(dataset)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

        sigmas = int(dataset_params.get("sigmas",3))
        thetas = int(dataset_params.get("thetas", 4))
        frequencies = dataset_params.get("frequencies", "0.05#0.25").split("#")

        kernels = []

        for theta in range(thetas):

            theta = theta / thetas * np.pi
            for sigma in (1, sigmas):
                for frequency in frequencies:
                    frequency = float(frequency)
                    kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x = sigma, sigma_y = sigma))
                    kernels.append(kernel)

        self.kernels = kernels

    def compute_feats(self,image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        img = x.permute(1,2,0)
        resized_img = img.cpu().numpy()
        resized_img = np.array((resized_img*256),np.uint8)

        gabor = [self.compute_feats(resized_img[:, :, i], self.kernels).reshape((1,-1)) for i in range(resized_img.shape[2])]

        artificial_x = np.concatenate(gabor)
        return torch.from_numpy(artificial_x).float(), y, x

    def __len__(self):
        return self.tensors[0].size(0)

class SIFTTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, dataset=None, data_dir: str = './data', dataset_size=32, nb_keypoints=30):

        if dataset is not None:
            if dataset=="cifar10_train":
                transform_chain = transforms.Compose([transforms.ToTensor(), transforms.Resize((dataset_size,dataset_size))])
                dataset = datasets.CIFAR10(root=data_dir,
                                           train=True,
                                           transform=transform_chain,
                                           download=True)

                tensors = _load_dataset(dataset)

        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.nb_keypoints = nb_keypoints
        self.sift = cv.SIFT_create()

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        img = x.permute(1,2,0)
        resized_img = img.cpu().numpy()

        gray = cv.cvtColor(np.array((resized_img*256),np.uint8), cv.COLOR_RGB2GRAY)
        kp = self.sift.detect(gray, None)
        mat = [[a.pt[0],a.pt[1],a.angle] for a in kp]
        size = [a.size for a in kp]
        zipped_keypoints = sorted(zip(size, mat))
        sift_x = [keypoint for _, keypoint in zipped_keypoints]

        artificial_x = sift_x[:self.nb_keypoints]
        while(len(artificial_x)<self.nb_keypoints):
            artificial_x.append([0,0,0])

        return torch.from_numpy(np.array(artificial_x)).float(), y, x

    def __len__(self):
        return self.tensors[0].size(0)
