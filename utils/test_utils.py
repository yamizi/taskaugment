import os
import pickle
import pprint
import json
from os.path import exists, join

import numpy as np
import torch
from torchvision import transforms
import sklearn.metrics
import sklearn, sklearn.model_selection
from utils.attacks.pgd import MTPGD
from utils.attacks.aa import MTAA
from utils.xrayvision import tqdm, ml_to_mt
from utils.imgtransformation_tasks import jigsaw, rotation
from utils.xrayvision import init_seed


def build_output(model, device, t, images, targets, criterion, task_outputs, task_targets,avg_loss, pathologies=None):
    with torch.no_grad():
        if isinstance(model, torch.nn.Module):
            outputs = model(images)
        else:
            outputs = model.forward(images)

        if not isinstance(outputs, dict):
            if hasattr(model,"pathologies"):
                pathologies = [f"class_object#{t}" for t in model.pathologies]

            outputs = ml_to_mt(outputs, pathologies)

        #if isinstance(outputs, dict):
        #    outputs_list = [v for (k, v) in outputs.items() if "class_object" in k]
        #    outputs = torch.cat(outputs_list, 1)

        loss = torch.zeros(1).to(device).double()

        target_labels = list(targets.keys())
        output_labels = list(outputs.keys())
        labels = [e for e in output_labels if e in target_labels]

        for task in labels:
        #for task in range(targets.shape[1]):
            task_output = outputs.get(task).squeeze()
            task_target = targets.get(task).squeeze()

            if task_output is None or task_target is None:
                continue
            mask = ~torch.isnan(task_target)
            task_output = task_output[mask]
            #shape = mask.shape[2:] if len(mask.shape)==4 else mask.shape[1:]
            #task_output = transforms.Resize(shape)(task_output)[mask] if task in ["depth","hog","ae"] else task_output[mask] #.squeeze(1) if len(task_output[mask].shape)>1 else task_output[mask]
            task_target = task_target[mask]
            if len(task_target) > 0:
                loss += criterion(task_output.double(), task_target.double())

            task_outputs[task] = np.concatenate([task_outputs[task], task_output.detach().cpu().numpy()])
            if task_targets is not None:
                task_targets[task] = np.concatenate([task_targets[task], task_target.detach().cpu().numpy()])

        loss = loss.sum()
        avg_loss.append(loss.detach().cpu().mean().numpy())
        t.set_description(f'Loss = {np.mean(avg_loss):4.4f}')

    return task_targets, task_outputs, avg_loss


def test(model, dataset, cfg, experiment=None, limit=None, load_weights=True, model_surrogate=None, surrogates=None):
    print("Our config:")
    pprint.pprint(cfg)
    xp_name = cfg.dataset + "-" + cfg.model + "-" + cfg.uniqueid
    record_roc = cfg.record_roc if hasattr(cfg,"record_roc") else 0

    os.makedirs(join(cfg.output_dir, "metrics"),exist_ok=True)
    os.makedirs(join(cfg.output_dir, "gradients"), exist_ok=True)

    device = 'cuda' if cfg.cuda else 'cpu'
    if not torch.cuda.is_available() and cfg.cuda:
        device = 'cpu'
        print("WARNING: cuda was requested but is not available, using cpu instead.")

    print(f'Using device: {device}')

    if not exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    
    # Setting the seed
    init_seed(cfg)


    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads, 
                                               pin_memory=cfg.cuda)

    criterion = torch.nn.BCEWithLogitsLoss()

    # loading model weights
    if load_weights:
        weights_file = os.path.join(cfg.output_dir, cfg.weights_file)
        weights = torch.load(weights_file).state_dict()

        if "_src" in weights_file:
            weights = dict(zip(model.state_dict().keys(), weights.values()))

        if "-avg-" in weights_file:
            from torch.optim.swa_utils import AveragedModel
            model = AveragedModel(model)

        model.load_state_dict(weights)
        print("Target model weights loaded: {0}".format(weights_file))

        if model_surrogate is not None:
            weights_file = os.path.join(cfg.output_dir, cfg.weights_file_surrogate)
            weights = torch.load(weights_file).state_dict()

            if "_src" in weights_file:
                weights = dict(zip(model_surrogate.state_dict().keys(), weights.values()))

            model_surrogate.load_state_dict(weights)
            print("Surrogate model weights loaded: {0}".format(weights_file))

        else:
            model_surrogate = model

    model.to(device)
    model.eval()

    model_surrogate.to(device)
    model_surrogate.eval()

    attack_tasks = cfg.attack_target.split("-")
    if attack_tasks[0]:

        if "AA" in cfg.algorithm:
            atk = MTAA(model, attack_tasks, eps=cfg.max_eps / 255,
                       n_classes=data_loader.dataset.nb_classes['class_object#multilabel'])
        else:
            criteria = {k:criterion for k in attack_tasks}
            atk = MTPGD(model_surrogate, attack_tasks, criteria, eps=cfg.max_eps / 255, alpha=cfg.step_eps / 255, steps=cfg.steps
                        ,record_all_tasks=cfg.record_all_tasks, random_start=cfg.random_start)
    else:
        atk = None

    avg_loss = []
    avg_advloss = []
    task_outputs={}
    task_advoutputs={}
    task_targets={}
    task_aucs = {}
    task_accs = {}
    tasks_fpr = {}
    tasks_tpr = {}
    tasks_metrics = {}
    task_advaccs = {}
    task_advaucs = {}
    task_gradients = {}
    task_cosine = []
    task_cov = []


    tasks_eval = [f"class_object#{t}" for t in data_loader.dataset.pathologies]

    if not isinstance(dataset.nb_classes, dict):
        dataset.nb_classes = {k:dataset.nb_classes for k in tasks_eval}

    if cfg.loss_rot:
        tasks_eval.append("rotation")
    if cfg.loss_jigsaw:
        tasks_eval.append("jigsaw")

    if cfg.loss_depth:
        tasks_eval.append("depth")

    if cfg.loss_ae:
        tasks_eval.append("ae")

    if cfg.loss_hog:
        tasks_eval.append("hog")

    if hasattr(model,"model"):
        tasks_eval = [t for t in tasks_eval if t in model.model.tasks]

    for i, task in enumerate(tasks_eval):
        task_outputs[task] = np.array([])
        task_advoutputs[task] = np.array([])
        task_targets[task] = np.array([])
        task_aucs[task] = []
        task_accs[task] = []
        task_advaucs[task] = []
        task_advaccs[task] = []
        tasks_fpr[task] = []
        tasks_tpr[task] = []
        tasks_metrics[task] = []

    t = tqdm(data_loader)
    for batch_idx, samples in enumerate(t):

        if limit and (batch_idx > limit):
            print("breaking out")
            break

        images = samples["img"].to(device)
        targets = samples["lab"].to(device)

        if not isinstance(targets, dict):
            targets = ml_to_mt(targets, tasks_eval)

        if cfg.loss_jigsaw:
            images, target_jig = jigsaw(images, cfg.permutation)
            target_jig = torch.nn.functional.one_hot(target_jig.long(), cfg.permutations_jigsaw).to(device)
            targets["jigsaw"] = target_jig
            dataset.nb_classes["jigsaw"] = target_jig.shape[1]


        if cfg.loss_rot:
            images, target_rot = rotation(images)
            target_rot = target_rot.to(device)
            target_rot = torch.nn.functional.one_hot(target_rot.long(), cfg.nb_rotations).to(device)
            targets["rotation"] = target_rot
            dataset.nb_classes["rotation"] = target_rot.shape[1]

        if cfg.loss_ae:
            task = "autoencoder" if samples.get("autoencoder").shape[1] == 3 else "autoencoder1c"
            targets[task] = images.to(device)

        if cfg.loss_hog:
            targets["hog"] = samples["hog"].to(device)

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
                targets["depth"] = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=sample.shape[2:],
                    mode="bicubic"
                )

        if "macro" in cfg.labelfilter:
            targets["class_object#macro"] = samples["macro"].to(device)

        if atk:
            adv_images = atk(images, targets)
            if cfg.record_all_tasks:

                if cfg.record_all_tasks > -1 and batch_idx % cfg.record_all_tasks==0:
                    pkl_file = join(cfg.output_dir, "gradients", f"{xp_name}-gradients-b{batch_idx}.pkl")
                    torch.save(atk.gradients, pkl_file)
                    if cfg.record_gradients_assets and experiment is not None:
                        experiment.log_asset(open(pkl_file, "rb"),f"gradients-b{batch_idx}.pkl",copy_to_tmp = False)
                        print(f"gradients saved to {pkl_file}")
                    pkl_file = join(cfg.output_dir, "gradients", f"{xp_name}-labels-b{batch_idx}.pkl")
                    torch.save(atk.labels, pkl_file)

                if experiment is not None:
                    for (step_i,cov) in enumerate(atk.gradients_cov):
                        experiment.log_metric("gradients_covariance",cov,step=step_i, epoch=batch_idx)

                        experiment.log_metric("gradients_dot", atk.gradients_dot[step_i], step=step_i,
                                              epoch=batch_idx)
                        experiment.log_metric("gradients_curve", atk.gradients_curve[step_i], step=step_i,
                                              epoch=batch_idx)
                        experiment.log_metric("gradients_magnitude", atk.gradients_magn[step_i], step=step_i,
                                              epoch=batch_idx)

                        if cfg.record_gradients_assets:
                            pkl_file = join(cfg.output_dir, "gradients", f"{xp_name}-cosine-b{batch_idx}.pkl")
                            torch.save(atk.gradients_cosine, pkl_file)
                            experiment.log_asset(open(pkl_file, "rb"), f"cosine-b{batch_idx}.pkl", copy_to_tmp=False)

        with torch.no_grad():

            task_targets, task_outputs, avg_loss = build_output(model, device, t, images, targets, criterion, task_outputs,
                                                                task_targets, avg_loss, pathologies=tasks_eval)

            if atk:
                _, task_advoutputs, avg_advloss = build_output(model, device, t, adv_images, targets, criterion, task_advoutputs,
                                                                None, avg_advloss)

            target_labels = list(task_targets.keys())
            output_labels = list(task_outputs.keys())
            labels = [e for e in output_labels if e in target_labels]

            for task in labels:
                batch_target = task_targets[task]
                if len(np.unique(batch_target))!= 1:
                    if dataset.nb_classes.get(task,2)==2:
                        target = batch_target
                        output = task_outputs[task]
                        adv_output = task_advoutputs[task]
                    else:
                        target = batch_target.reshape(-1,dataset.nb_classes[task])
                        output = task_outputs[task].reshape(-1,dataset.nb_classes[task])
                        adv_output = task_advoutputs[task].reshape(-1,dataset.nb_classes[task])

                    if task in ["depth","hog","ae"]:
                        task_aucs[task].append(np.nan)

                    elif len(np.unique(batch_target)) > 1:
                        task_auc = sklearn.metrics.roc_auc_score(batch_target, task_outputs[task])
                        task_aucs[task].append(task_auc)
                    else:
                        task_aucs[task].append(np.nan)

                    if "class_object" in task or task=="jigsaw" or task=="rotation":
                        if len(output.shape)==1:
                            task_accuracy = sklearn.metrics.accuracy_score(target, torch.sigmoid(torch.tensor(output))>0.5)
                        else:
                            task_accuracy =sklearn.metrics.accuracy_score(target.argmax(1), output.argmax(1))
                        task_accs[task].append(task_accuracy)

                    if record_roc:
                        fpr, tpr, _ = sklearn.metrics.roc_curve(target, output)
                        tasks_fpr[task].append(fpr)
                        tasks_tpr[task].append(tpr)

                        metrics = sklearn.metrics.precision_recall_fscore_support(target, output>0.5)
                        tasks_metrics[task].append(metrics)
                    if atk:

                        if "class_object" in task:
                            if len(adv_output.shape) == 1:
                                task_accuracy = sklearn.metrics.accuracy_score(target, adv_output>0.5)
                            else:
                                task_accuracy = sklearn.metrics.accuracy_score(target.argmax(1), adv_output.argmax(1))
                            task_advaccs[task].append(task_accuracy)

                        if task in ["depth", "hog", "ae"]:
                            task_advaucs[task].append(np.nan)

                        elif len(np.unique(batch_target)) > 1:
                            task_advauc = sklearn.metrics.roc_auc_score(batch_target, task_advoutputs[task])
                            task_advaucs[task].append(task_advauc)
                else:
                    task_aucs[task].append(np.nan)
                    task_accs[task].append(np.nan)
                    if atk:
                        task_advaucs[task].append(np.nan)
                        task_advaccs[task].append(np.nan)

    experiment.log_metric("loss", np.mean(avg_loss))
    if atk:
        experiment.log_metric("advloss", np.mean(avg_advloss))

    for (task, task_auc) in task_aucs.items():

        task_label = task.replace("class_object#","")
        outputs = json.dumps(task_outputs[task].tolist())
        experiment.log_asset_data(outputs,f"{task_label}_outputs.json")

        outputs = json.dumps(task_targets[task].tolist())
        experiment.log_asset_data(outputs, f"{task_label}_targets.json")

        if task not in ["depth", "hog", "ae"]:
            task_auc = np.asarray(task_auc)
            auc = task_auc[~np.isnan(task_auc)][-1]
            print(f'Task {task_label} Avg AUC = {auc:4.4f} ')
            experiment.log_metric(f"{task_label}_auc",auc)

        if "class_object" in task:
            task_acc = np.asarray(task_accs[task])
            acc = task_acc[~np.isnan(task_acc)][-1]
            print(f'Task {task_label} Avg Accuracy = {acc:4.4f} ')
            experiment.log_metric(f"{task_label}_acc", acc)

        #experiment.log_metric(f"{task_label}_fpr", tasks_fpr[task][-1])
        #experiment.log_metric(f"{task_label}_tpr", tasks_tpr[task][-1])
        if record_roc:
            experiment.log_asset_data(json.dumps([a.tolist() for a in tasks_metrics[task][-1]]),
                                      f"{task_label}_metrics.json")
            experiment.log_asset_data(json.dumps(tasks_fpr[task][-1].tolist()),
                                      f"{task_label}_fpr.json")
            experiment.log_asset_data(json.dumps(tasks_tpr[task][-1].tolist()),
                                      f"{task_label}_tpr.json")

        if atk:
            if task not in ["depth", "hog", "ae"]:
                task_advauc = np.asarray(task_advaucs[task])
                advauc = task_advauc[~np.isnan(task_advauc)][-1]
                print(f'Task {task_label}  Avg ADV AUC = {advauc:4.4f} ')
                experiment.log_metric(f"{task_label}_advauc", advauc)

            if "class_object" in task:
                task_advacc = np.asarray(task_advaccs[task])
                acc = task_advacc[~np.isnan(task_advacc)][-1]
                print(f'Task {task_label} Avg ADV Accuracy = {acc:4.4f} ')
                experiment.log_metric(f"{task_label}_advacc", acc)
        else:
            task_advaucs = None

    metrics = {
        "weights": cfg.weights_file,
        "avg_loss": avg_loss,
        "task_aucs": task_aucs,
        "task_accs": task_accs,
        "avg_advloss": avg_advloss,
        "task_advaccs": task_advaccs,
        'task_advaucs': task_advaucs
    }

    metrics_file = f'{xp_name}-testmetrics.pkl'
    with open(join(cfg.output_dir, "metrics",metrics_file), 'wb') as f:
        pickle.dump(metrics, f)

        experiment.log_others(metrics)

    return task_aucs,  avg_loss, task_advaucs,  avg_advloss