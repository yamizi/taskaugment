import sys, os
sys.path.append(".")

from comet_ml import Experiment
import torch

import argparse
from experiments.aaai22.model_transfer import run as run_model_transfer
from experiments.aaai22.domain_knowledge import run as run_domain_knowledge
from experiments.aaai22.rq1_force1 import runner as run_rq1_force1
from experiments.aaai22.rq1_forceall import runner as run_rq1_forceall
from experiments.aaai22.rq1_forbid1 import runner as run_rq1_forbid1
from experiments.aaai22.rq1_sortab import runner as run_rq1_sortab
from experiments.aaai22.rq1_sortall import runner as run_rq1_sortall

parser = argparse.ArgumentParser(description='experiments runner')
parser.add_argument('--criterion', type=str, default="bce")
parser.add_argument('--dataset', type=str, default="NIH")
parser.add_argument('--data_folder', type=str, default="")
parser.add_argument('--model', type=str, default="xrayvision")
parser.add_argument('--model_labels', type=str, default="")
parser.add_argument('--lib', type=str, default="torchattack")
parser.add_argument('--algorithm', default="pgd")
parser.add_argument('--workspace', default="aaai22_health")
parser.add_argument('--reduction', default="none")
parser.add_argument('--eps_step', type=float, default=0.001)
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--batch_size',type=int, default=16)
parser.add_argument('--num_random_init',type=int, default=1)
parser.add_argument('--max_eps',type=str, default="2")
parser.add_argument('--nb_batches',type=int, default=0)
parser.add_argument('--steps', type=int, default=25)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--norm', type=str, default="Linf")
parser.add_argument('--name', type=str, default="model_transfer")
parser.add_argument('--experiment', type=str, default="model_transfer")
parser.add_argument('--store_examples', type=int, default=0)
args = parser.parse_args()

data_folders= {"NIH":"./data/NIH Chest X-rays", "CHEX":"D://datasets//CheXpert//test.csv##D://datasets//CheXpert"}


if __name__ == '__main__':
    vals = vars(args)
    parameters = dict([(k,v.rstrip()) if isinstance(v,str) else (k,v) for (k,v) in vals.items()])

    model_labels = parameters.get("model_labels")
    name = parameters.get("name")
    parameters["max_eps"]= float(parameters.get("max_eps",1/255))
    dataset = parameters.get("dataset")
    data_folder= parameters.get("data_folder") if parameters.get("data_folder")!="" else data_folders.get(dataset)
    data_folder = data_folder.split("##")
    if len(data_folder)==1:
        data_folder = data_folder[0]
    else:
        data_folder = {"csv":data_folder[0],"img":data_folder[1]}

    parameters["data_folder"] = data_folder

    if model_labels == "":
        model_labels = None
    else:
        model_labels = model_labels.split("##")

    experiment = parameters.get("experiment")
    if experiment=="model_transfer":
        run_model_transfer(parameters, name=name, model_labels=model_labels)

    elif experiment=="domain_knowledge":

        combinations = [{"criterion": "ord_ce"}, {"criterion": "ord"},
                        {"criterion": "mse"}, {"criterion": "bce"},
                        {"criterion":"sortedsampled_ce"},{"criterion":"sortedsampled"}]

        directions = ["direct", "indirect"]
        directions = ["direct"]
        for comb in combinations:
            for d in directions:
                parameters = {**parameters, **comb}
                parameters["direction"] = d
                run_domain_knowledge(parameters, name=name, model_labels=model_labels)

    elif "threat_" in experiment:
        for nm in model_labels:
            parameters["model_name"] = nm
            if "force1" in experiment:
                f = run_rq1_force1
            elif "forceall" in experiment:
                f = run_rq1_forceall
            elif "forbid1" in experiment:
                f = run_rq1_forbid1
            elif "sortab" in experiment:
                f = run_rq1_sortab
            else:
                f = run_rq1_sortall
            f(parameters, "{}_{}".format(experiment,name))



