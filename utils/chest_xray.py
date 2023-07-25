import os
import torch
import torchxrayvision as xrv
from torchvision import transforms
from utils.datasets import split_dataset

def get_model(device="cuda", shuffle = False, data_path='NIH Chest X-rays', model_ckpt="stage4_1e-05_12.pth",
              batch_size=32):

    device = torch.device(device)
    data_dir = os.path.join('data', data_path)

    from NIH_Chest_X_Rays.datasets import XRaysTestDataset
    from NIH_Chest_X_Rays import config

    XRayTest_dataset = XRaysTestDataset(data_dir, transform=config.transform)
    test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size=batch_size, shuffle=shuffle)
    """
    if loss_func == 'FocalLoss':  # by default
        from losses import FocalLoss
        loss_fn = FocalLoss(device=device, gamma=2.).to(device)
    elif loss_func == 'BCE':
        loss_fn =   torch.nn.BCEWithLogitsLoss().to(device)
    """

    ckpt = torch.load(os.path.join(config.models_dir, model_ckpt))

    model = ckpt['model']
    model.to(device)
    model.eval()


    return test_loader, model


def get_xrayvision(device="cuda", shuffle = False, data_folder='NIH Chest X-rays', model_ckpt="densenet121-res224-all",
              batch_size=32, dataset="NIH", split=False, parameters=None, return_loader=False):

    image_size = 224
    model_ckpt = "densenet121-res224-all" if model_ckpt==None else model_ckpt

    if "jfhealthcare" in model_ckpt:
        model = xrv.baseline_models.jfhealthcare.DenseNet()
    elif "chexpert" in model_ckpt:
        model = xrv.baseline_models.chexpert.DenseNet("D:\models\chexpert_weights.zip")

    elif "resnet50" in model_ckpt:
        model = xrv.models.ResNet(weights="resnet50-res512-all")
    else:

        if model_ckpt=="densenet121-res224-nb":
            model_ckpt="densenet121-res224-mimic_nb"

        if model_ckpt=="densenet121-res224-ch":
            model_ckpt="densenet121-res224-mimic_nb"
        model = xrv.models.DenseNet(weights=model_ckpt)


    if parameters.get("dataset_dir"):
        from argparse import Namespace
        from utils.xrayvision import init_dataset
        config = Namespace(**parameters)

        train_dataset, test_dataset = init_dataset(config, image_size)

        if return_loader:
            train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        if split:
            loader =  train_dataset, test_dataset
        else:
            loader =  test_dataset

    else:

        transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(224)])

        imgpath = os.path.join('data', data_folder) if isinstance(data_folder, str) else data_folder["img"]
        dataset = dataset.lower()
        if(dataset=="nih"):
            imgpath = data_folder if (data_folder[0]=="/" or data_folder[0]=="~") else os.path.join(data_folder,"images","1")
            print(imgpath)
            XRayTest_dataset = xrv.datasets.NIH_Dataset(imgpath=imgpath,transform=transform)

        elif (dataset == "kaggle"):
            XRayTest_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(imgpath=imgpath,
                                                           transform=transform)

        elif (dataset == "chex"):

            XRayTest_dataset = xrv.datasets.CheX_Dataset(imgpath=data_folder["img"],
                                               csvpath=data_folder["csv"],
                                               transform=transform)
        elif (dataset == "nih2"):
            XRayTest_dataset = xrv.datasets.NIH_Google_Dataset(imgpath=imgpath)

        elif (dataset == "pc"):
            XRayTest_dataset = xrv.datasets.PC_Dataset(imgpath=imgpath)

        xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies, XRayTest_dataset)

        if split:
            loader = split_dataset(dataset, split, batch_size, len(XRayTest_dataset.labels), shuffle=shuffle)
        else:
            loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size=batch_size, shuffle=shuffle)

    model.to(device)
    model.eval()

    return loader, model


