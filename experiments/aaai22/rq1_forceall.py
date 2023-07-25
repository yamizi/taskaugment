import sys
sys.path.append(".")

from comet_ml import Experiment
import torch
import random
import numpy as np
import traceback
from utils import init_comet
from utils.models import get_model_dataset
from utils.attacks import PGDL

from utils.losses import ForceTopClasses
from utils.metrics import success_force_all
from utils.metrics import common_success_metrics

def run(parameters, name="topk_accuracy", random_seed=0, experiment=None, topk=(1,3)):
    random.seed(random_seed)
    print("running", parameters)
    device = torch.device(parameters.get("device", "cuda"))

    loader, m = get_model_dataset(parameters, device=parameters.get("device", "cuda"), return_loader=True)

    if experiment is None and name is not None:
        experiment = init_comet(args=parameters, project_name=name)

    nb_batches = parameters.get("nb_batches", 0)

    attack = PGDL(m, eps=parameters.get("max_eps"), steps=parameters.get("max_iter"), random_start=False)

    for i, batch in enumerate(loader):
        if nb_batches is not None and nb_batches > 0 and i >= nb_batches:
            break

        print("batch ",i)
        direction = parameters.get("direction", "asc")
        loss = ForceTopClasses(criterion=parameters.get("criterion", None),experiment=experiment, direct=direction)

        if isinstance(batch, dict):
            imgs = batch['img'].to(device)
            labels = batch['lab'].to(device)
            labels[torch.isnan(labels)] = 0
        else:
            imgs = batch[0].to(device)
            labels = batch[1].to(device)

        with torch.no_grad():
            clean_output = m(imgs)

        nb_inputs = clean_output.shape[0]
        nb_classes = clean_output.shape[1]

        if parameters.get("target")=="random":
            perm = torch.cat([torch.randperm(nb_classes).unsqueeze(0) for i in range(nb_inputs)])
            classes_to_forbid = perm[:,0:topk[0]]
        elif parameters.get("target")=="bottom":
            _, bottomk_values = clean_output.topk(topk[0], 1, False, True)
            classes_to_forbid = bottomk_values

        attack.set_mode_targeted()
        adv = attack(imgs, classes_to_forbid, loss=loss)

        with torch.no_grad():
            output = m(adv)
            _topk = range(topk[0], topk[1]+1)
            force_acc = success_force_all(output,torch.transpose(classes_to_forbid,0,1).to(output.device), _topk)
            for i, acc in enumerate(force_acc):
                experiment.log_metric("force_success_top{}".format(_topk[i]), acc)

            common_success_metrics(experiment, clean_output, output, labels, None)

        print(acc)


def runner(parameters, name="rq1_forceall", topk=None):
    print("ForceAll")
    topk = parameters.get("topk",(3,5)) if topk is None else topk
    combinations = [{"criterion": "ord_ce"}, {"criterion": "ord"}, {"criterion": "mse"}, {"criterion": "bce"},
                    {"criterion": "sortedsampled_ce"}, {"criterion": "sortedsampled"}]
    targets = ["random", "bottom"]
    directions = ["asc", "desc"]
    directions = ["desc"]
    for comb in combinations:
        for t in targets:
            for d in directions:
                parameters = {**parameters, **comb}
                parameters["target"] = t
                parameters["direction"] = d
                run(parameters, name=name, topk=topk)


if __name__ == '__main__':
    parameters = {"use_hidden": False, "pretrained": True, "criterion": "mse", "algorithm": "PGD", "max_eps": 8/255,
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.05, "num_random_init": 1, "batch_size": 32,
                  "nb_batches":4, "lib": "torchattack", "model": "cifar10_resnet20", "dataset": "cifar10",
                  "workspace":"SortedAttacks", "target":"random", "topk":(3,5)}

    runner(parameters)
