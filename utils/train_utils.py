import copy
import os
import pickle
import pprint
from glob import glob
from os.path import exists, join

import numpy as np
import torch
import sklearn.metrics
import sklearn, sklearn.model_selection
import torchvision.transforms

import torchxrayvision as xrv
from utils.attacks.pgd import MTPGD
from utils.xrayvision import tqdm, ml_to_mt
from utils.multitask_losses import get_criteria, MaxupCrossEntropyLoss

from utils.imgtransformation_tasks import jigsaw, rotation
from utils.xrayvision import init_seed
from utils.weights import init_strategy

from utils.data.cifar10_utils import (upper_limit, lower_limit, std, clamp)
from utils.models import update_bn
from torchvision import transforms

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def train(model, dataset, cfg, experiment=None, load_weights=False, surrogates=None):
    print("Our config:")
    pprint.pprint(cfg)

    dataset_name = cfg.dataset + "-" + cfg.model + "-" + cfg.name
    weights_strategy = None
    weight_args = {}

    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    if cfg.labelfilter:
        dataset_name = dataset_name + "-" + cfg.labelfilter

    if hasattr(cfg, 'labelsource') and cfg.labelsource:
        dataset_name = dataset_name + "_src-" + cfg.labelsource

    if hasattr(cfg, 'strategy') and cfg.strategy:
        dataset_name = dataset_name + "_str-" + cfg.strategy

    if cfg.loss_sex:
        dataset_name = dataset_name + "_sex-" + str(cfg.loss_sex)

    if cfg.loss_hog:
        dataset_name = dataset_name + "_hog-" + str(cfg.loss_hog)

    if cfg.loss_ae:
        dataset_name = dataset_name + "_ae-" + str(cfg.loss_ae)

    if cfg.loss_age:
        dataset_name = dataset_name + "_age-" + str(cfg.loss_age)

    if cfg.loss_depth:
        dataset_name = dataset_name + "_depth-" + str(cfg.loss_depth)

    if cfg.loss_jigsaw:
        dataset_name = dataset_name + "_jigsaw-" + str(cfg.loss_jigsaw) + "-" + str(cfg.permutations_jigsaw)

    if cfg.loss_rot:
        dataset_name = dataset_name + "_rot-" + str(cfg.loss_rot) + "-" + str(cfg.nb_rotations)

    #if cfg.loss_detect:
    #    dataset_name = dataset_name + "_detect-" + str(cfg.loss_detect)

    if cfg.augment:
        dataset_name = dataset_name + "_augment-" + str(cfg.augment)

    if cfg.cutmix:
        dataset_name = dataset_name + "_cutmix"

    if cfg.attack_target:
        dataset_name = dataset_name + "_adv-" + cfg.attack_target

    if cfg.optimizer:
        dataset_name = dataset_name + "_optim-" + cfg.optimizer

    if cfg.data_subset < 1:
        dataset_name = dataset_name + "_subset-" + str(cfg.data_subset)

    if cfg.force_cosine !="":
        dataset_name = dataset_name + "_cosine-" + str(cfg.force_cosine)

    #if cfg.img_size !=32:
    #    dataset_name = dataset_name + "_size-" + str(cfg.img_size)

    if cfg.weight_strategy != "":

        if cfg.weight_strategy != "_":
            dataset_name = dataset_name + "_w-" + str(cfg.weight_strategy)

        weights_strategy, weight_args = init_strategy(cfg, model, dataset, device=device)

    if cfg.algorithm != "PGD" and cfg.algorithm != "":
        dataset_name = dataset_name + "_a-" + str(cfg.algorithm)

    print(cfg.output_dir)
    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # Setting the seed
    init_seed(cfg)

    # loading model weights
    if load_weights:
        weights_file = os.path.join(cfg.output_dir, cfg.weights_file)
        weights = torch.load(weights_file).state_dict()

        if cfg.fine_tune==1:
            weights = {k: weights[k] for k in model.state_dict().keys() if k in weights.keys() if 'fc' not in k}
            weights = {**model.state_dict(), **weights}

        if "_src" in weights_file:
            weights = dict(zip(model.state_dict().keys(), weights.values()))
        elif "pretrain" in weights_file or "posttrain" in cfg.output_dir:
            weights = {k:weights[k] for k in model.state_dict().keys() if k in weights.keys()}
            weights = {**model.state_dict(), **weights}

        model.load_state_dict(weights)
        print("Target model weights loaded: {0}".format(weights_file))

    # Dataset
    if hasattr(dataset, "csv") and "patientid" in dataset.csv.columns:
        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=cfg.seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    else:
        gss = sklearn.model_selection.ShuffleSplit(train_size=0.8, test_size=0.2, random_state=cfg.seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset))))

    train_dataset = xrv.datasets.SubsetDataset(copy.deepcopy(dataset), train_inds)
    valid_dataset = xrv.datasets.SubsetDataset(copy.deepcopy(dataset), test_inds)

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size//cfg.maxup,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads,
                                               pin_memory=cfg.cuda)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads,
                                               pin_memory=cfg.cuda)

    # Checkpointing
    start_epoch = 0
    best_metric = 0.
    weights_for_best_validmetric = None
    swa_scheduler = None
    swa_model = None
    metrics = []
    weights_path = join(cfg.output_dir, f'{dataset_name}-e*.pt')
    weights_files = glob(weights_path)  # Find all weights files
    if len(weights_files) and not cfg.restart:
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(cfg.output_dir, f'{dataset_name}-e')):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max()
        weights_file = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()][0]
        print(f"weight file: {weights_file}")
        checkpoint = torch.load(weights_file, map_location='cpu')
        model.load_state_dict(checkpoint.state_dict())

        metric_file = join(cfg.output_dir, f'{dataset_name}-metrics.pkl')

        if exists(metric_file):
            print("loading metrics from {}".format(metric_file))
            with open(metric_file, 'rb') as f:
                metrics = pickle.load(f)

            best_metric = metrics[-1]['best_metric']

        weights_for_best_validmetric = model.state_dict()

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    else:
        print("Starting training from epoch 0 for {0}.".format(weights_path))
    # Optimizer
    if "adam" in cfg.optimizer:
        optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5, amsgrad=True)
        scheduler = None
    else:
        optim = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

        lr_steps = cfg.num_epochs * len(train_loader)

        if cfg.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=cfg.lr_min, max_lr=cfg.lr * 2,
                                                          step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif cfg.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                             gamma=0.1)
        elif cfg.lr_schedule == 'cosine':
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optim,
                lr_lambda=lambda step: cosine_annealing(
                    step + start_epoch * len(train_loader),
                    cfg.num_epochs * len(train_loader),
                    1,  # since lr_lambda computes multiplicative factor
                    cfg.lr_min / cfg.lr))
        else:
            scheduler = None # torch.optim.lr_scheduler.ConstantLR(total_iters=1)

    if "swa" in cfg.optimizer:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_scheduler = SWALR(optim, swa_lr=0.01, anneal_epochs=len(train_loader)*5)
        swa_model = AveragedModel(model)

    train_criterion = torch.nn.BCEWithLogitsLoss() if cfg.maxup==1 else MaxupCrossEntropyLoss(cfg.maxup, cfg.batch_size)
    valid_criterion = torch.nn.BCEWithLogitsLoss()

    loss_criteria, loss_tasks = get_criteria()

    model.to(device)

    params = vars(cfg)
    attack_tasks = params.get("attack_target", "").split("-")
    max_eps = int(params.get("max_eps", 8))
    step_eps = int(params.get("step_eps", 2))
    steps = int(params.get("steps", 4))
    random_start = bool(params.get("random_start", 1))

    if attack_tasks[0]:
        aux_tasks = ["jigsaw", "rotation","age","sex","depth","hog","autoencoder","autoencoder1c"]
        attack_tasks = [f"class_object#{t}" if t not in aux_tasks else t for t in attack_tasks]
        criteria = {k: loss_criteria.get(k,valid_criterion) for k in attack_tasks}
        atk = MTPGD(model, attack_tasks, criteria, eps=max_eps / 255, alpha=step_eps / 255, steps=steps,
                    random_start=random_start, algorithm=cfg.algorithm)
    else:
        atk = None
    for epoch in range(start_epoch, cfg.num_epochs):

        if "FAST" in cfg.algorithm:
            avg_loss, losses = train_epoch_fast(cfg=cfg,
                                       epoch=epoch,
                                       model=model,
                                       device=device,
                                       optimizer=optim,
                                       train_loader=train_loader,
                                       criterion=train_criterion,
                                       scheduler=scheduler,
                                       limit=params.get("limit"),
                                       weights_strategy=weights_strategy, weight_args=weight_args,
                                       attack_tasks=attack_tasks,
                                        start_epoch=start_epoch,
                                       swa_scheduler = swa_scheduler,
                                       swa_model= swa_model,
                                       loss_criteria=loss_criteria,
                                       surrogates=surrogates)
        else:
            avg_loss, losses = train_epoch(cfg=cfg,
                                       epoch=epoch,
                                       model=model,
                                       device=device,
                                       optimizer=optim,
                                       train_loader=train_loader,
                                       criterion=train_criterion,
                                       scheduler=scheduler,
                                       atk=atk, limit=params.get("limit"),
                                       weights_strategy=weights_strategy, weight_args=weight_args,
                                       algorithm=cfg.algorithm,
                                       swa_scheduler = swa_scheduler,
                                        start_epoch=start_epoch,
                                       swa_model = swa_model,
                                       loss_criteria=loss_criteria,
                                       surrogates=surrogates)

        if swa_scheduler:
            update_bn(train_loader, swa_model, device=device)

        metric_valid, _, _, _, loss_valid = valid_test_epoch(name='Valid',
                                                          epoch=epoch,
                                                          model=swa_model if swa_scheduler else model,
                                                          device=device,
                                                          data_loader=valid_loader,
                                                          criterion=valid_criterion, limit=params.get("limit"),
                                                          main_metric=params.get("main_metric", "auc"),
                                                         surrogates=surrogates)

        if np.mean(metric_valid) > best_metric:
            best_metric = np.mean(metric_valid)
            if swa_scheduler:
                weights_for_best_validmetric = swa_model.state_dict()
                save_path = join(cfg.output_dir, f'{dataset_name}-avg-best.pt')
                torch.save(weights_for_best_validmetric, save_path)
            else:
                weights_for_best_validmetric = model.state_dict()
                save_path = join(cfg.output_dir, f'{dataset_name}-best.pt')
                torch.save(model, save_path)
            print("Saving best model to {0}.".format(save_path))
            if experiment is not None:
                experiment.log_parameter("save_path", save_path)
            # only compute when we need to

        stat = {
            "epoch": epoch + 1,
            "trainloss": avg_loss,
            "validmetric": metric_valid,
            "validloss": loss_valid.detach().cpu(),
            'best_metric': best_metric,
        }

        if atk is not None:
            stat = {**stat,
                    **{f"robust_trainacc_{k.replace('class_object#', '')}": torch.mean(torch.stack(v)) for (k, v) in
                       atk.robust_accs.items()}}
            atk.reset_acc()

        for (k, v) in losses.items():
            if v and len(v):
                if isinstance(k, int):
                    k = train_loader.dataset.pathologies[k]
                stat["{}_loss".format(k)] = np.mean(v)

        metrics.append(stat)
        if experiment is not None:
            experiment.log_metrics(stat, step=epoch + 1)
            if hasattr(scheduler, "get_lr"):
                experiment.log_metric("learning_rate", scheduler.get_lr(), step=epoch + 1)

        with open(join(cfg.output_dir, f'{dataset_name}-metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        if epoch < cfg.save_all or epoch % cfg.save_skip == 0:
            checkpoint_path = join(cfg.output_dir, f'{dataset_name}-e{epoch + 1}.pt')
            torch.save(model, checkpoint_path)
            if experiment is not None:
                experiment.log_parameter("save_path_checkpoint", checkpoint_path)


    return metrics, best_metric, weights_for_best_validmetric


def init_weights(cfg, train_loader, device):
    weights_jigsaw = weights_rotation = weights_hog = weights_ae = weights_sex = weights_age = weight_depth = 1
    weights_dict = dict()
    negative_weights_dict = dict()
    if cfg.labelfilter != "" and cfg.taskweights:
        if train_loader.dataset.is_binary:
            weights = np.nansum(train_loader.dataset.labels, axis=0)
        else:
            weights = np.count_nonzero(~np.isnan(train_loader.dataset.labels), 0)
        weights = weights.max() - weights + weights.mean()
        weights = weights / weights.max()
        weights = torch.from_numpy(weights).to(device).float() if weights is np.ndarray else weights
        weights_dict = dict(zip(train_loader.dataset.pathologies, weights))
        negative_weights = [True if e[0] == "#" else False for e in cfg.labelfilter.split("-")]
        negative_weights_dict = dict(zip(train_loader.dataset.pathologies, negative_weights))
        weights_age = 1 / 10
        weights_sex = 1 / 50
        weights_ae = 1 / 500
        weights_hog = 1 / 25

        print("task weights", weights, negative_weights)

    return weights_jigsaw, weights_rotation, weights_hog, weights_ae, weights_sex, weights_age, weight_depth, weights_dict, negative_weights_dict


def init_losses():

    avg_losses = {"sex": [], "age": [], "hog": [], "ae": [], "rotation": [], "jigsaw": [], "cosine": [], "raw_cosine":[],
                  "depth":[]}
    avg_loss = []

    return avg_losses, avg_loss

def init_batch_labels(cfg, model, samples, device, train_loader, surrogates=None):
    labels = {}
    images = samples["img"]
    images = images.float().to(device)
    targets = samples["lab"].to(device)

    if cfg.smoothed_std:
        # augment inputs with noise
        images = images + torch.randn_like(images, device=device) * cfg.smoothed_std

    if cfg.cutmix:
        from utils.cutmix import cutmix
        images, targets = cutmix(images,targets)

    if cfg.maxup>1:
        from utils.cutmix import cutmix
        images = torch.repeat_interleave(images, repeats=cfg.maxup, dim=0)
        targets = torch.repeat_interleave(targets, repeats=cfg.maxup, dim=0)
        images, targets = cutmix(images, targets)


    if cfg.loss_jigsaw:
        images, target_jig = jigsaw(images, cfg.permutation)
        target_jig = torch.nn.functional.one_hot(target_jig.long(), cfg.permutations_jigsaw).to(device)
        labels["jigsaw"] = target_jig.to(device)
        train_loader.dataset.pathologies.append("jigsaw")

    if cfg.loss_rot:
        images, target_rot = rotation(images)
        target_rot = target_rot.to(device)
        target_rot = torch.nn.functional.one_hot(target_rot.long(), cfg.nb_rotations).to(device)
        labels["rotation"] = target_rot.to(device)
        train_loader.dataset.pathologies.append("rotation")

    if cfg.loss_ae:
        task = "autoencoder" if samples.get("autoencoder").shape[1]==3 else "autoencoder1c"
        labels[task] = images.to(device)

    if cfg.loss_hog:
        labels["hog"] = samples["hog"].to(device)

    if cfg.loss_depth:
        with torch.no_grad():
            sample = images.to(device)
            if device == "cuda":
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            depth_model = surrogates["depth"]
            net_w, net_h = 256, 256

            depth_transform = transforms.Resize((net_w, net_h))
            prediction = depth_model.forward(depth_transform(sample))
            labels["depth"] = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=sample.shape[2:],
                    mode="bicubic"
                )

    if "macro" in cfg.labelfilter:
        labels["class_object#macro"] = samples["macro"].to(device)

    #if cfg.loss_detect:
    #    labels["class_object#detector"] = samples["detector"].to(device)

    tasks_list = model.model.tasks if hasattr(model, "model") else model.tasks
    if tasks_list is None:
        tasks =  [f"class_object#{t}" for t in train_loader.dataset.pathologies]
    else:
        tasks = [f"class_object#{t}" for t in train_loader.dataset.pathologies if f"class_object#{t}" in tasks_list]
    labels = {**ml_to_mt(targets, tasks), **labels}

    return labels, tasks, images

def batch_losses(cfg, model, labels, outputs, adv_outputs, algorithm, images, criterion, device ,avg_losses, samples, loss_criteria, raw_outputs, weights, batch_index=None ):
    losses_list = []
    beta = 0.5
    rand_binary = np.random.rand()

    loss = torch.zeros(1).to(device).float()
    weights_jigsaw, weights_rotation, weights_hog, weights_ae, weights_sex, weights_age, weights_depth, weights_dict, negative_weights_dict = weights

    for (task, output) in outputs.items():

        task_label = task.replace("class_object#", "")
        task_output_ = outputs[task].squeeze()
        task_target_ = labels[task].squeeze()
        mask = ~torch.isnan(task_target_)
        task_output = task_output_[mask].to(device)
        task_target = task_target_[mask].to(device)
        if len(task_target) > 0:

            if "TRADES" in algorithm:
                from utils.multitask_losses import TRADES_loss
                beta = 6.0  # regularization, i.e., 1/lambda in TRADES
                batch_size = len(images)
                adv_output = adv_outputs[task].squeeze()
                clean_loss = criterion(task_output.float(), task_target.float())
                adv_loss = (1.0 / batch_size) * TRADES_loss(adv_output.to(device), task_output_.to(device), mask)
                task_loss = clean_loss + beta * adv_loss

            elif "MADRY" in algorithm:

                adv_output = adv_outputs[task].squeeze()[mask].to(device)
                clean_loss = criterion(task_output.float(), task_target.float())
                adv_loss = criterion(adv_output.float(), task_target.float())

                if "ALTERNATE" in algorithm:
                    task_loss = clean_loss if batch_index%2==0 else adv_loss
                elif "RANDOM" in algorithm:
                    task_loss = clean_loss if rand_binary<beta else adv_loss
                else:
                    task_loss = (1 - beta) * clean_loss + beta * adv_loss

            else:
                task_loss = criterion(task_output.float(), task_target.float())

            losses_list.append(task_loss)
            if cfg.taskweights:
                weighted_loss = weights_dict.get(task_label, 1) * task_loss
                if "log" in cfg.name:
                    weighted_loss = torch.log(weighted_loss)

                loss += -weighted_loss if negative_weights_dict.get(task_label, False) else weighted_loss
            else:
                loss += task_loss

            if avg_losses.get(task, None) is None:
                avg_losses[task] = [task_loss.detach().cpu().numpy()]
            else:
                avg_losses[task].append(task_loss.detach().cpu().numpy())


    # printing the image
    # plt.imshow(images[0].detach().cpu().permute(1, 2, 0))

    if cfg.loss_sex:
        sex = samples.get("sex")
        loss_fn = loss_criteria["sex"]
        sex_loss = torch.log(loss_fn(raw_outputs["sex"].float().to(device), sex.float().to(device)))
        loss += weights_sex * float(cfg.loss_sex) * sex_loss
        avg_losses["sex"].append(float(cfg.loss_sex) * sex_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_sex) * sex_loss)

    if cfg.loss_age:
        age = samples.get("age")
        loss_fn = loss_criteria["age"]
        age_loss = loss_fn(raw_outputs["age"].float().to(device), age.float().to(device))
        loss += weights_age * float(cfg.loss_age) * torch.log(age_loss)
        avg_losses["age"].append(float(cfg.loss_age) * torch.log(age_loss.detach().cpu()).numpy())
        losses_list.append(float(cfg.loss_age) * age_loss)

    if cfg.loss_hog:

        lbl_ = samples["hog"].to(device)
        out_ = raw_outputs["hog"].to(device)

        hog_loss = loss_criteria["hog"](torchvision.transforms.Resize(lbl_.shape[-1])(out_),lbl_)
        loss += weights_hog * float(cfg.loss_hog) * hog_loss
        avg_losses["hog"].append(float(cfg.loss_hog) * hog_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_hog) * hog_loss)
        #plt.imshow(samples["hog"][0].cpu().permute(1, 2, 0))

    if cfg.loss_ae:
        task = "autoencoder" if samples.get("autoencoder").shape[1]==3 else "autoencoder1c"

        lbl_ = samples[task].to(device)
        out_ = raw_outputs[task].to(device)

        ae_loss = loss_criteria[task](torchvision.transforms.Resize(lbl_.shape[-1])(out_),lbl_)
        loss += weights_ae * float(cfg.loss_ae) * ae_loss
        avg_losses["ae"].append(float(cfg.loss_ae) * ae_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_ae) * ae_loss)

    if cfg.loss_jigsaw:
        loss_fn = loss_criteria["jigsaw"]
        jigsaw_loss = loss_fn(raw_outputs["jigsaw"].float().to(device), labels["jigsaw"].float().to(device))
        loss += weights_jigsaw * float(cfg.loss_jigsaw) * jigsaw_loss
        avg_losses["jigsaw"].append(float(cfg.loss_jigsaw) * jigsaw_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_jigsaw) * jigsaw_loss)

    if cfg.loss_rot:
        loss_fn = loss_criteria["rotation"]
        rot_loss = loss_fn(raw_outputs["rotation"].float().to(device), labels["rotation"].float().to(device))
        loss += weights_rotation * float(cfg.loss_rot) * rot_loss
        avg_losses["rotation"].append(float(cfg.loss_rot) * rot_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_rot) * rot_loss)

    if cfg.loss_depth:

        loss_fn = loss_criteria["depth"]

        lbl_ = labels["depth"].to(device)
        out_ = raw_outputs["depth"].to(device)

        depth_loss = loss_fn(torchvision.transforms.Resize(lbl_.shape[-1])(out_),lbl_)
        loss += weights_depth * float(cfg.loss_depth) * depth_loss
        avg_losses["depth"].append(float(cfg.loss_depth) * depth_loss.detach().cpu().numpy())
        losses_list.append(float(cfg.loss_depth) * depth_loss)



    #if cfg.loss_detect:
    #    loss_fn = loss_criteria["class_object"]
    #    detect_loss = loss_fn(raw_outputs["class_object#detector"].float().to(device), labels["class_object#detector"].float().to(device))
    #    loss += float(cfg.loss_detect) * detect_loss
        #avg_losses["detector"].append(float(cfg.loss_detect) * detect_loss.detach().cpu().numpy())
    #    losses_list.append(float(cfg.loss_detect) * detect_loss)

    if cfg.force_cosine and int(cfg.force_cosine) > 0:
        ### Use simplification from https://arxiv.org/pdf/2104.09937.pdf (Gradient Matching)
        from utils.gradients import record_task_gradients
        gradients, outputs = record_task_gradients(model, False, images, labels, keep_graph=True, return_outputs=True)
        criterion = torch.nn.CosineSimilarity()
        task1 = list(gradients.values())[0].to(device)#.flatten().to(device)
        task2 = list(gradients.values())[1].to(device)#.flatten().to(device)

        task_1 = task1.reshape(task1.shape[0],-1)
        task_2 = task2.reshape(task2.shape[0], -1)

        summ = task_1 + task_2
        diff = task_1 - task_2
        norm_1 = task_1.norm()
        norm_2 = task_2.norm()
        """
         dot_product = torch.dot(task_1, task_2)
        cos_angle = dot_product / (norm_1 * norm_2)
        curvature_measure = (1 - cos_angle ** 2) * diff.norm() ** 2 / summ.norm() ** 2
        """

        cosine_sim = criterion(task_1, task_2)
        #cosine_loss = (1 - torch.logsumexp(cosine_sim,0) ** 2) * diff.norm() ** 2 / summ.norm() ** 2
        cosine_loss = (1 - cosine_sim ** 2) * diff.norm(dim=1) ** 2 / summ.norm(dim=1) ** 2
        cosine_loss = cosine_loss.mean()
        #cosine_loss = torch.logsumexp(cosine_sim.abs(), 0)
        # cosine_loss = float(cfg.force_cosine) * (cosine_sim + 1) / 2

        from utils.losses import Houdini
        if int(cfg.force_cosine)==1:
            cosine_loss = Houdini.apply(task1, task2, cosine_loss)

        elif int(cfg.force_cosine)==2:
            out = torch.exp(torch.nn.functional.log_softmax(outputs['class_object#multilabel']))
            label = labels['class_object#multilabel']
            cosine_loss = Houdini.apply(out, label, cosine_loss)

        #else:

        cosine_loss = torch.abs(cosine_loss)

        if not torch.isnan(cosine_loss):
            loss += cosine_loss
            avg_losses["raw_cosine"].append(cosine_sim.mean().detach().cpu().numpy())
            avg_losses["cosine"].append(cosine_loss.detach().cpu().numpy())
            losses_list.append(cosine_loss)
        else:
            avg_losses["raw_cosine"].append(cosine_sim.mean().detach().cpu().numpy())
            avg_losses["cosine"].append(cosine_sim.mean().detach().cpu().numpy())
            print("nan gradient")


    loss = loss.sum()

    if cfg.featurereg:
        feat = model.features(images)
        loss += feat.abs().sum()

    if cfg.weightreg:
        loss += model.classifier.weight.abs().sum()


    return losses_list, loss


def train_epoch(cfg, epoch, model, device, train_loader, optimizer, criterion, scheduler=None, limit=None, atk=None,
                weights_strategy=None, weight_args={}, algorithm="PGD", swa_scheduler=None, swa_model=None,
                start_epoch=0, loss_criteria=dict(),surrogates=None):
    model.train()

    weights = init_weights(
        cfg, train_loader, device)
    avg_losses, avg_loss = init_losses()

    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):

        if limit and (batch_idx > limit):
            print("breaking out")
            break

        optimizer.zero_grad()
        labels, tasks, images = init_batch_labels(cfg, model, samples, device, train_loader, surrogates)
        adv_outputs = None

        if atk:
            adv_images = atk(images, labels)
            if "TRADES" in algorithm or "MADRY" in algorithm:
                adv_outputs = model(adv_images)
                adv_outputs = {k: adv_outputs[k] for (i, (k, v)) in enumerate(labels.items()) if "class_object" in k}
            #elif cfg.loss_detect:
            #    if  labels.get("class_object#detector") is not None:
            #        mask = labels["class_object#detector"].squeeze()[:,1].bool()
            #        mask_matrix = mask.repeat((3, 32, 32, 1)).movedim(3, 0)
            #        images = torch.where(mask_matrix, images, adv_images)
                    #0 = adversarial image
            else:
                images = adv_images

        raw_outputs = model(images)

        if not isinstance(raw_outputs,dict) and hasattr(model,"tasks"):
            if model.tasks is None:
                #raw_outputs = ml_to_mt(raw_outputs, labels.keys())
                raw_outputs = ml_to_mt(raw_outputs, labels.keys())
            elif len(model.tasks)==1:
                raw_outputs = {model.tasks[0]:raw_outputs}
            else:
                raw_outputs = ml_to_mt(raw_outputs,model.tasks )

        outputs = {k: raw_outputs[k] for (i, (k, v)) in enumerate(labels.items()) if "class_object" in k}

        losses_list, loss = batch_losses(cfg, model, labels, outputs, adv_outputs, algorithm, images, criterion, device, avg_losses,
                     samples, loss_criteria, raw_outputs, weights, batch_index=batch_idx)

        avg_loss.append(loss.detach().cpu().numpy())

        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        if losses_list and len(losses_list)>0:
            if weights_strategy is None:
                loss.backward()
            else:
                #print("custom weighing strategy {} with {} losses".format(cfg.weight_strategy,len(losses_list)))
                weights_strategy.backward(losses_list, **weight_args)

            optimizer.step()
            if scheduler is not None:

                if swa_scheduler is not None:
                    if epoch > cfg.swa_start:
                        swa_model.update_parameters(model)
                        swa_scheduler.step()
                    else:
                        scheduler.step()
                else:
                    scheduler.step()

    return np.mean(avg_loss), avg_losses


def train_epoch_fast(cfg, epoch, model, device, train_loader, optimizer, criterion, attack_tasks, scheduler=None, limit=None,
                weights_strategy=None, weight_args={}, swa_scheduler=None, swa_model = None,start_epoch=0,loss_criteria=dict(), surrogates=None):
    model.train()

    weights = init_weights(
        cfg, train_loader, device)
    avg_losses, avg_loss = init_losses()

    t = tqdm(train_loader)
    for batch_idx, samples in enumerate(t):

        if limit and (batch_idx > limit):
            print("breaking out")
            break


        labels, tasks, images = init_batch_labels(cfg, model, samples, device, train_loader)
        X = images
        y = labels
        delta = torch.zeros_like(X).to(device)
        epsilon = (cfg.max_eps / 255.) / std
        alpha = (cfg.step_eps / 255.) / std

        for j in range(len(epsilon)):
            delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)

        delta.requires_grad = True
        output = model(X + delta[:X.size(0)])
        loss = [criterion(output[task].float(), y[task].float()) for task in attack_tasks]
        loss = torch.stack(loss).sum()
        loss.backward()

        grad = delta.grad.detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
        delta = delta.detach()

        adv_outputs = None
        images = X + delta[:X.size(0)]
        optimizer.zero_grad()

        raw_outputs = model(images)
        outputs = {k: raw_outputs[k] for (i, (k, v)) in enumerate(y.items()) if "class_object" in k}

        losses_list, loss = batch_losses(cfg, model, labels, outputs, adv_outputs, "PGD", images, criterion, device, avg_losses,
                     samples, loss_criteria, raw_outputs, weights, batch_index=batch_idx)

        avg_loss.append(loss.detach().cpu().numpy())

        t.set_description(f'Epoch {epoch + 1} - Train - Loss = {np.mean(avg_loss):4.4f}')

        weights_strategy.backward(losses_list, **weight_args)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return np.mean(avg_loss), avg_losses

def valid_test_epoch(name, epoch, model, device, data_loader, criterion, limit=None, main_metric="auc", surrogates=None):
    model.eval()

    avg_loss = []
    task_outputs = {}
    task_targets = {}
    for task in range(data_loader.dataset[0]["lab"].shape[0]):
        task_outputs[task] = []
        task_targets[task] = []

    with torch.no_grad():
        t = tqdm(data_loader)
        for batch_idx, samples in enumerate(t):

            loss = torch.zeros(1).to(device).double()

            if limit and (batch_idx > limit):
                print("breaking out")
                break

            images = samples["img"].to(device)
            raw_outputs = model(images)

            if data_loader.dataset.dataset.type=="xray":
                targets = samples["lab"].to(device)
            else:
                main_task = "class_object#multilabel" if "class_object#multilabel" in raw_outputs.keys() else ("class_object#macro" if "class_object#macro" in raw_outputs.keys() else ("jigsaw" if "jigsaw" in raw_outputs.keys() else "rotation"))
                targets = samples["lab"].to(device) if main_task == "class_object#multilabel" else (samples["macro"].to(device) if main_task == "class_object#macro" else None)

            nb_tasks = targets.shape[1] if targets is not None else 0

            if torch.all(torch.isnan(targets)):
                continue
            binary_outputs = True
            if isinstance(raw_outputs, dict):
                outputs_list = [v for (k, v) in raw_outputs.items() if "class_object" in k]
                #outputs_list =  outputs_list + [v for (k, v) in raw_outputs.items() if k in aux_tasks]
                if len(outputs_list):
                    binary_outputs = len(outputs_list[0].shape) == 1
                    outputs = torch.cat(outputs_list, 1) if binary_outputs else outputs_list
                else:
                    outputs = outputs_list
            else:
                list_pathologies = model.pathologies if hasattr(model,"pathologies") else model.tasks
                if list_pathologies is None:
                    outputs = raw_outputs
                else:
                    labels_mask = [m.replace('class_object#','') in data_loader.dataset.pathologies for m in list_pathologies]
                    outputs = raw_outputs[:, labels_mask]


            for task in range(nb_tasks):
                if binary_outputs:
                    task_output = outputs[:, task]
                    task_target = targets[:, task]
                else:
                    task_output = outputs[task]
                    task_target = targets[:, task]

                mask = ~torch.isnan(task_target)
                task_target = task_target[mask]
                task_output = task_output[mask].reshape_as(task_target)
                if len(task_target) > 0:
                    loss += criterion(task_output.double(), task_target.double())

                task_outputs[task].append(task_output.detach().cpu().numpy())
                task_targets[task].append(task_target.detach().cpu().numpy())

            loss = loss.sum()

            avg_loss.append(loss.detach().cpu().numpy())
            t.set_description(f'Epoch {epoch + 1} - {name} - Loss = {np.mean(avg_loss):4.4f}')


        if len(task_targets[task]):
            for task in range(nb_tasks):
                task_outputs[task] = np.concatenate(task_outputs[task])
                task_targets[task] = np.concatenate(task_targets[task])

        task_aucs = []
        for task in range(nb_tasks):
            if main_metric == "auc":
                if len(np.unique(task_targets[task])) > 1:
                    task_auc = sklearn.metrics.roc_auc_score(task_targets[task], task_outputs[task])
                    # print(task, task_auc)
                    task_aucs.append(task_auc)
                else:
                    task_aucs.append(np.nan)
            elif main_metric == "acc":
                if isinstance(data_loader.dataset.nb_classes, dict):
                    nb_classes = list(data_loader.dataset.nb_classes.values())[task]
                else:
                    nb_classes = data_loader.dataset.nb_classes

                targets_acc = task_targets[task].reshape(-1, nb_classes)
                outputs_acc = task_outputs[task].reshape(-1, nb_classes)
                task_accuracy = sklearn.metrics.accuracy_score(targets_acc.argmax(1), outputs_acc.argmax(1))
                task_aucs.append(task_accuracy)

    task_aucs = np.asarray(task_aucs)
    auc = np.mean(task_aucs[~np.isnan(task_aucs)]) if len(task_aucs) else 0
    print(f'Epoch {epoch + 1} - {name} - Avg Metric = {auc:4.4f}')

    return auc, task_aucs, task_outputs, task_targets, loss
