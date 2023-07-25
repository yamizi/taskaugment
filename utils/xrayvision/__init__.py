import copy
import random
import numpy as np
import torch
import torchvision
import sklearn, sklearn.model_selection

import torchxrayvision as xrv
from utils.multitask_models import resnet50_taskonomy as MTResnet50
from utils.multitask_models import resnet34_taskonomy as MTResnet34


from tqdm import tqdm as tqdm_base

def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def ml_to_mt(labels, tasks):
    return dict(zip(tasks, torch.transpose(labels,0,1)))

def init_seed(cfg):
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    if cfg.cuda:
        torch.cuda.manual_seed_all(cfg.seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

def init_tasks(cfg, tasks_labels, dataset=None):
    from utils.multitask_losses import get_criteria
    auxiliary_tasks = ["hog","autoencoder","depth","jigsaw","rotation","sex","age"]
    _, supported_tasks = get_criteria()
    if tasks_labels is None:
        tasks = [f"class_object#{dataset.pathologies[i]}" for i in range(dataset.labels.shape[1]) if not dataset.pathologies[i] in auxiliary_tasks]
    elif len(tasks_labels) > 1:
        tasks = [pathology if pathology in supported_tasks else f"class_object#{pathology}" for pathology in tasks_labels]
    else:
        if tasks_labels[0] != "":
            task = tasks_labels[0].replace("class_object#", "")
            tasks = [f"class_object#{task}"]
        else:
            tasks = []

    surrogates = {}
    if cfg.loss_sex != 0:
        tasks.append("sex")

    if cfg.loss_age != 0:
        tasks.append("age")

    if cfg.loss_hog != 0:
        tasks.append("hog")

    if cfg.loss_ae != 0:
        if cfg.dataset in ['chex', 'nih', "pc"]:
            tasks.append("autoencoder1c")
        else:
            tasks.append("autoencoder")

    if cfg.loss_jigsaw != 0:
        tasks.append("jigsaw")

    if cfg.loss_rot != 0:
        tasks.append("rotation")

    if cfg.loss_detect != 0:
        tasks.append("class_object#detector")

    if cfg.loss_depth != 0:
        tasks.append("depth")

        from MiDaS.midas.midas_net_custom import MidasNet_small
        from torchvision.transforms import Compose

        depth_model = MidasNet_small("data/midas_v21_small-70d6b9c8.pt", features=64, backbone="efficientnet_lite3",
                                     exportable=True,
                                     non_negative=True, blocks={'expand': True})
        depth_model.eval()
        if cfg.cuda:
            depth_model = depth_model.to(memory_format=torch.channels_last)
            depth_model = depth_model.half()
            depth_model.to("cuda")

        surrogates["depth"] = depth_model


    num_classes = [cfg.output_size, {"permutations_jigsaw" : cfg.permutations_jigsaw,"nb_secondary_labels":cfg.nb_secondary_labels, "nb_rotations":cfg.nb_rotations}]
    tasks = list( dict.fromkeys(tasks) )

    return tasks, num_classes, surrogates

def init_model(cfg, train_dataset, pathologies=None):
    tasks = None
    surrogates = None
    num_classes = train_dataset.labels.shape[1]

    if "multi_task" in cfg.model:
        tasks, num_classes, surrogates = init_tasks(cfg, pathologies, train_dataset)

    if "densenet" in cfg.model:
        model = xrv.models.DenseNet(num_classes=num_classes, in_channels=1,tasks=tasks,
                                    **xrv.models.get_densenet_params(cfg.model))

    elif "resnet101" in cfg.model:
        model = torchvision.models.resnet101(tasks=tasks, num_classes=num_classes, pretrained=False)
        #patch for single channel
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "resnet50" in cfg.model:
        model = MTResnet50(tasks=tasks, num_classes=num_classes)
        model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "resnet34" in cfg.model:
        model = MTResnet34(tasks=tasks, num_classes=num_classes)
        model.encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    elif "shufflenet_v2_x2_0" in cfg.model:
        model = torchvision.models.shufflenet_v2_x2_0(tasks=tasks, num_classes=num_classes, pretrained=False)
        model.conv1[0] = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
    elif "squeezenet1_1" in cfg.model:
        model = torchvision.models.squeezenet1_1(tasks=tasks, num_classes=num_classes, pretrained=False)
        #patch for single channel
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    else:
        raise Exception("no model")

    if cfg.cuda:
        model = model.to("cuda")
    return model, surrogates

def init_dataset(cfg, size=512):
    data_aug = None
    if cfg.data_aug:
        data_aug = torchvision.transforms.Compose([
            xrv.datasets.ToPILImage(),
            torchvision.transforms.RandomAffine(cfg.data_aug_rot,
                                                translate=(cfg.data_aug_trans, cfg.data_aug_trans),
                                                scale=(1.0 - cfg.data_aug_scale, 1.0 + cfg.data_aug_scale)),
            torchvision.transforms.ToTensor()
        ])

    transforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(size)])

    datas = []
    datas_names = []

    if "nih" in cfg.dataset:
        dataset = xrv.datasets.NIH_Dataset(
            imgpath=cfg.dataset_dir + "/NIH/images-224",
            csvpath=cfg.dataset_dir + "/NIH/Data_Entry_2017.csv",
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA"])
        datas.append(dataset)
        datas_names.append("nih")
    if "pc" in cfg.dataset:
        dataset = xrv.datasets.PC_Dataset(
            imgpath=cfg.dataset_dir + "/PC/images-224",
            csvpath=cfg.dataset_dir + "/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA"])
        datas.append(dataset)
        datas_names.append("pc")
    if "chex" in cfg.dataset:
        dataset = xrv.datasets.CheX_Dataset(
            imgpath=cfg.dataset_dir + "/CheXpert",
            csvpath=cfg.dataset_dir + "/CheXpert/train.csv",
            transform=transforms, data_aug=data_aug, unique_patients=False)
        datas.append(dataset)
        datas_names.append("chex")
    if "google" in cfg.dataset:
        dataset = xrv.datasets.NIH_Google_Dataset(
            imgpath=cfg.dataset_dir + "/images-512-NIH",
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("google")
    if "mimic_ch" in cfg.dataset:
        dataset = xrv.datasets.MIMIC_Dataset(
            imgpath="/scratch/users/joecohen/data/MIMICCXR-2.0/files/",
            csvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-chexpert.csv.gz",
            metacsvpath=cfg.dataset_dir + "/MIMICCXR-2.0/mimic-cxr-2.0.0-metadata.csv.gz",
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA"])
        datas.append(dataset)
        datas_names.append("mimic_ch")
    if "openi" in cfg.dataset:
        dataset = xrv.datasets.Openi_Dataset(
            imgpath=cfg.dataset_dir + "/OpenI/images/",
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("openi")
    if "rsna" in cfg.dataset:
        dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
            imgpath=cfg.dataset_dir + "/kaggle-pneumonia-jpg/stage_2_train_images_jpg",
            transform=transforms, data_aug=data_aug, unique_patients=False, views=["PA"])
        datas.append(dataset)
        datas_names.append("rsna")
    if "siim" in cfg.dataset:
        dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
            imgpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/dicom-images-train",
            csvpath=cfg.dataset_dir + "/SIIM_TRAIN_TEST/train-rle.csv",
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("siim")
    if "vin" in cfg.dataset:
        dataset = xrv.datasets.VinBrain_Dataset(
            imgpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train",
            csvpath=cfg.dataset_dir + "vinbigdata-chest-xray-abnormalities-detection/train.csv",
            transform=transforms, data_aug=data_aug)
        datas.append(dataset)
        datas_names.append("vin")

    print("datas_names", datas_names)

    if cfg.labelunion:
        newlabels = set()
        for d in datas:
            newlabels = newlabels.union(d.pathologies)
        newlabels.remove("Support Devices")
        print(list(newlabels))
        for d in datas:
            xrv.datasets.relabel_dataset(list(newlabels), d)
    else:
        labelfilter = cfg.labelfilter.replace("#","")
        label_filters = labelfilter.split("-") if len(cfg.labelfilter) > 0 else xrv.datasets.default_pathologies
        for d in datas:
            xrv.datasets.relabel_dataset(label_filters, d)

    # cut out training sets
    train_datas = []
    test_datas = []
    for i, dataset in enumerate(datas):

        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]

        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=cfg.seed)

        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_inds = train_inds[0:int(cfg.data_subset*len(train_inds))]
        dataset.set_includes(cfg)
        train_dataset = xrv.datasets.SubsetDataset(copy.deepcopy(dataset), train_inds)
        test_dataset = xrv.datasets.SubsetDataset(copy.deepcopy(dataset), test_inds)

        # disable data augs
        test_dataset.data_aug = None

        train_datas.append(train_dataset)
        test_datas.append(test_dataset)

    if len(datas) == 0:
        raise Exception("no dataset")
    elif len(datas) == 1:
        train_dataset = train_datas[0]
        test_dataset = test_datas[0]
    else:
        print("merge datasets")
        train_dataset = xrv.datasets.Merge_Dataset(train_datas)
        test_dataset = xrv.datasets.Merge_Dataset(test_datas)

    print("train_dataset.labels.shape", train_dataset.labels.shape)
    print("test_dataset.labels.shape", test_dataset.labels.shape)
    print("train_dataset", train_dataset)
    print("test_dataset", test_dataset)

    train_dataset.type = "xray"
    test_dataset.type = "xray"

    return train_dataset, test_dataset