import sys
import os
sys.path.append(".")

from comet_ml import Experiment
import torch
import random
import numpy as np
import traceback
from utils import init_comet
from utils.models import get_model_dataset
from utils.attacks import PGDL

from utils.losses import ForbidTopClasses
from utils.metrics import success_forbid
from utils.metrics import common_success_metrics

def run(parameters, name="topk_accuracy", random_seed=0, experiment=None, topk=(1,3,5)):
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

        direction = parameters.get("nb_batches", "asc")
        loss = ForbidTopClasses(criterion=parameters.get("criterion", None),experiment=experiment, direct=direction)

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
            classes_to_forbid = torch.randint(0, nb_classes, (nb_inputs,1))
        elif parameters.get("target")=="top":
            topk_values = clean_output.topk(topk[0], 1, True, True)
            classes_to_forbid = topk_values[1]

        attack.set_mode_targeted()
        adv = attack(imgs, classes_to_forbid, loss=loss)

        with torch.no_grad():
            output = m(adv)
            forbid_acc = success_forbid(output,classes_to_forbid.squeeze(1).to(output.device),topk)
            for i,acc in enumerate(forbid_acc):
                experiment.log_metric("forbid_success_top{}".format(topk[i]), acc)

            common_success_metrics(experiment, clean_output, output, labels, None)

        print(forbid_acc)


def runner(parameters,name="rq1_forbid",topk=(1, 3, 5)):
    print("Forbid1")
    combinations = [{"criterion": "ord_ce"}, {"criterion": "sortedsampled_ce"}, {"criterion": "ord"},
                    {"criterion": "sortedsampled"},
                    {"criterion": "mse"}, {"criterion": "bce"}]

    #combinations = [{"criterion": "ord_ce"}, {"criterion": "ord"}, {"criterion": "mse"}, {"criterion": "bce"}]
    targets = ["random", "top"]
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
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.05, "num_random_init": 1, "batch_size": 16,
                  "nb_batches":8, "lib": "torchattack", "model": "cifar10_resnet20", "dataset": "cifar10",
                  "workspace":"SortedAttacks", "target":"random"}

    #run(parameters, name="rq1_forbid", topk=(1,))
    #exit()

    parameters["data_folder"] = os.path.join("data", 'NIH Chest X-rays')
    parameters["dataset"] = "NIH"
    parameters["model"] ="xrayvision"

    runner(parameters)


