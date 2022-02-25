from __future__ import print_function, division
import os, os.path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class TaskonomyDataset(Dataset):
    """Face Landmarks dataset."""

    img_tasks = ["segmentsemantic", "depth_zbuffer", "normal"]
    numpy_tasks = ["class_object", "class_places"]
    img_x = ["rgb"]
    def __init__(self, root_dir, transform=None, normalize=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.normalize_input = normalize

        base_rgb = os.path.join(root_dir,self.img_x[0])
        self.path_images = [name.split("_domain")[0] for name in os.listdir(base_rgb) if os.path.isfile(os.path.join(base_rgb,name))]


    def __len__(self):
        return len(self.path_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t = self.img_x[0]
        img_x =  io.imread(os.path.join(self.root_dir,t,
                                "{}_domain_{}.png".format(self.path_images[idx],t)))

        img_names = [os.path.join(self.root_dir,t,
                                "{}_domain_{}.png".format(self.path_images[idx],t)) for t in self.img_tasks]
        images = [io.imread(img_name) for img_name in img_names]

        nps = [np.load(os.path.join(self.root_dir,t,
                                "{}_domain_{}.npy".format(self.path_images[idx],t))) for t in self.numpy_tasks]

        keys = self.img_x + self.img_tasks + self.numpy_tasks
        data = [img_x] + images + nps

        if self.transform:
            transformed = self.transform(dict(zip(keys, data)))
            sample = dict(zip(transformed.keys(), transformed.values()))

            if self.normalize_input:
                sample[t] = sample[t] * 2 - 1


        else:
            sample = dict(zip(keys, data))


        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample[TaskonomyDataset.img_x[0]]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        k, v = TaskonomyDataset.img_x[0], image
        sample[k] = transform.resize(v, (new_h, new_w))

        for k,v in sample.items():
            if k in TaskonomyDataset.img_tasks:
                sample[k] = transform.resize(v, (new_h, new_w))
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        keys = sample.keys()
        updated = dict([(k, torch.from_numpy(sample[k].transpose((2, 0, 1)))) if len(sample[k].shape)>2 else
                        (k, torch.from_numpy(sample[k])) for k in keys])

        return updated

if __name__ == "__main__":
    dataset = TaskonomyDataset("../taskonomy-sample-model-1", transform=transforms.Compose([
                                               Rescale(256),
                                               ToTensor()
                                           ]))
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)


    for i in range(len(dataset)):
        sample = dataset[i]

        print(i, sample['rgb'].shape)

        if i == 3:
            break

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['rgb'].size())

        if i_batch == 3:
            break
