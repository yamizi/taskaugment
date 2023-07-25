from torchvision import models
from torchvision import transforms

from collections import OrderedDict
from typing import Tuple
from torch import Tensor
import torch
from PIL import Image

import matplotlib.pyplot as plt

from utils.lenet import Net

from utils.resnet import ResNet
from utils.multitask_models import resnet18_taskonomy as MTResnet18
from utils.multitask_models import resnet50_taskonomy as MTResnet50
from utils.multitask_models import wideresnet_28_10 as MTWideResnet2810
from utils.multitask_models import wideresnet_70_16 as MTWideResnet7016

transformations = {
    "cifar10": {"std":[0.2471, 0.2435, 0.2616],"mean":[0.4914, 0.4822, 0.4465]},
    "cifar100": {"std":[0.2675, 0.2565, 0.2761],"mean":[0.5071, 0.4867, 0.4408]},
    "imagenet":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
    "coco":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
    "voc2012":{"std":[0.229, 0.224, 0.225],"mean":[0.485, 0.456, 0.406]},
    "taskonomy":{"mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]}
}

class Identity(torch.nn.Module):
    def __init__(self, activation=False):
        super(Identity, self).__init__()
        self.activation = activation

    def forward(self, x):
        if self.activation=="sigmoid":
            #[0,1]
            return torch.sigmoid(x)*2-1 #(1 + torch.sigmoid(x)) / 2
            #[-1,1]
            #return torch.sigmoid(x)*4-3
        elif self.activation=="relu":
            return torch.relu(x)
        else:
            return x


class ImageNormalizer(torch.nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: torch.nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float], device:str) -> torch.nn.Module:

    if "cuda" in device:
        device = "cuda:0"

    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return torch.nn.Sequential(layers).to(device)

def get_model_dataset(parameters, shuffle=False, device="cuda", return_loader=False, split=False, normalized=True):
    model = parameters.get("model", "")

    data_folder = parameters.get("data_folder", "~/datasets")
    batch_size = parameters.get("batch_size", 16)
    model_name = parameters.get("model_name")
    dataset_label = parameters.get("dataset", "aux_cifar10_train")
    if "train" in dataset_label:
        parameters["train"] = True

    if model.find("xrayvision") > -1:
        from utils.chest_xray import get_xrayvision
        dataset, model= get_xrayvision(device, shuffle=shuffle, batch_size=batch_size, dataset=dataset_label, model_ckpt=model_name,
                              data_folder=data_folder, split=split, parameters=parameters, return_loader=return_loader)
        if isinstance(dataset,tuple):
            dataset[0].type = "xray"
            dataset[1].type = "xray"
        else:
            dataset.type = "xray"

    elif model.find("retinopathy") > -1:
        from utils.retinopathy import get_model as retinopathy_model
        dataset, model= retinopathy_model(device, shuffle=shuffle, batch_size=batch_size)
        dataset.type = "other"

    elif model.find("housing") > -1:
        from utils.housing import get_model as housing_model
        dataset, model= housing_model(device, shuffle=shuffle, batch_size=batch_size)
        dataset.type = "other"

    else:
        from utils.datasets import get_dataset
        model, transform = get_model(parameters, shuffle=shuffle, device=device)
        dataset = get_dataset(parameters, shuffle=shuffle, return_loader=return_loader, split=split, transform=transform)
        dataset.type = "baseline"

    if normalized:
        dataset_split = dataset_label.split("_")
        dataset_transformations = transformations.get(dataset_split[1],transformations["imagenet"]) if len(dataset_split)>1 else transformations["imagenet"]
        model = normalize_model(model,dataset_transformations.get("mean"),dataset_transformations.get("std"),device=device
                                )

    return dataset, model


def get_model(parameters, shuffle=False, device="cuda"):
    model = parameters.get("model","")
    transform = None

    if model.find("imagenet")>-1:
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor() #,  # [4]
         ])

        if model.find("resnet18")>-1:
            model = models.resnet18(pretrained=parameters.get("pretrained"))
        elif model.find("wide_resnet50") > -1:
            model = models.wide_resnet50_2(pretrained=parameters.get("pretrained"))
        elif model.find("resnet50") > -1:
            model = models.resnet50(pretrained=parameters.get("pretrained"))

    elif model.find("cifar")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

        model = torch.hub.load("chenyaofo/pytorch-cifar-models", model, pretrained=True)

    elif model.find("coco")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])

        if model=="coco_v2":
            from utils.msc import MSC
            model = MSC(base=torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k',
                                        n_classes=182))
        elif model=="coco_yolov5":
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


    elif model.find("voc2012")>-1:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
            ])
        model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='voc12', n_classes=21)

    elif model.find("baseResnet") > -1:
        input_size = parameters.get("input_size", (1, 32))
        output_size = parameters.get("output_size", 10)
        model = ResNet(None, [2, 2,  2, 2], num_classes=output_size, input_channels=input_size[0], input_size=input_size[1])

    elif model.find("multi_task") > -1:

        output_size = parameters.get("output_size", 10)
        secondary_size = parameters.get("nb_secondary_labels",10)
        permutations_jigsaw = parameters.get("permutations_jigsaw", 10)
        nb_rotations = parameters.get("nb_rotations", 10)

        num_classes = [output_size, {"nb_secondary_labels":secondary_size, "permutations_jigsaw":permutations_jigsaw, "nb_rotations":nb_rotations}]
        pretrained = parameters.get("pretrained", 0)
        tasks = parameters.get("tasks")
        kwargs =  {"num_classes":num_classes,"img_size":parameters.get("img_size",256), "pretrained":pretrained}
        if model.find("resnet18") > -1:
            model = MTResnet18(tasks=tasks, **kwargs)
        elif model.find("resnet50") > -1:
            model = MTResnet50(tasks=tasks, **kwargs)
        elif model.find("wide2810") > -1:
            model = MTWideResnet2810(tasks=tasks, **kwargs)
        elif model.find("wideresnet") > -1:
            model = MTWideResnet7016(tasks=tasks, **kwargs)
    else:
        model = Net(parameters)

    if parameters.get("use_hidden")=="sigmoid_last":
        model.add_module("output",Identity(activation="sigmoid"))

    elif parameters.get("use_hidden"):
        model.fc = Identity()
        model.avgpool = Identity(activation="sigmoid")


    model.eval()
    if device=="cuda":
        model = model.cuda()

    return model, transform


def back_transform(image, dataset):
    info =transformations.get(dataset)
    image[:, 0, :, :] = image[:, 0, :, :] * info["std"][0] + info["mean"][0]
    image[:, 1, :, :] = image[:, 1, :, :] * info["std"][1] + info["mean"][1]
    image[:, 2, :, :] = image[:, 2, :, :] * info["std"][2] + info["mean"][2]
    return image

def forward_transform(image, dataset):
    info = transformations.get(dataset)
    image[:, 0, :, :] = (image[:, 0, :, :] - info["mean"][0]) / info["std"][0]
    image[:, 1, :, :] = (image[:, 1, :, :] - info["mean"][1]) / info["std"][1]
    image[:, 2, :, :] = (image[:, 2, :, :] - info["mean"][2]) / info["std"][2]
    return image


def compress_img(img, rate=75, format='jpeg', palette=256):
    from io import BytesIO

    byteImgIO = BytesIO()
    img.save(byteImgIO, format=format, quality=75)
    byteImgIO.seek(0)

    dataBytesIO = BytesIO(byteImgIO.read())
    compressed_img = Image.open(dataBytesIO)

    return compressed_img

def depth_img(img, depth=8):
    imageWithColorPalette = img.convert("P", palette=Image.ADAPTIVE, colors=depth)
    return imageWithColorPalette.convert("RGB")

def show_input_tensor(tensors, indices=(0,)):
    if tensors.shape[0]==1:
        tensors = tensors.squeeze(0)

    if len(tensors.shape) == 3 or len(tensors.shape) == 2:
        plt.imshow(tensors.cpu())
        plt.show()

    else:
        for index in indices:
            tensor = tensors[index].cpu()

            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            else:
                tensor = tensor.transpose(0,1).transpose(1,2)

            plt.imshow(tensor)
            plt.show()


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

    return models_differ

@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        elif isinstance(input, dict):
            input = input["img"]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
