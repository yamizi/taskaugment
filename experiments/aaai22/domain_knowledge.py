import sys, os
sys.path.append(".")

from comet_ml import Experiment
import torch
import random
import pandas as pd
from utils import init_comet
from scipy.special import softmax
from sklearn.preprocessing import normalize
import numpy as np
from utils.losses import OrdinalLoss

from utils.models import get_model_dataset
from utils.attacks import PGDL

from utils.losses import SortAllClasses
from utils.metrics import common_success_metrics


def get_less_cooccurred_target(batch, labels, map_target):
    batch_argmax = batch[:,:14].argmax(1)

    target = np.array([map_target[k.item()] for k in batch_argmax])
    target = torch.cat([torch.from_numpy(target), torch.zeros(target.shape[0], batch.shape[1] - target.shape[1])], 1)
    return target.to(batch.device)





def run_model(parameters, name="topk_accuracy", random_seed=0, model_labels = None, map_target=None ):
    random.seed(random_seed)
    print("running", parameters)

    models_labels = ["densenet121-res224-all", "resnet50","densenet121-res224-nih", "densenet121-res224-chex",
                     "densenet121-res224-pc", "densenet121-res224-mimic_nb", "densenet121-res224-mimic_ch",
                     "densenet121-res224-rsna"] if model_labels is None else model_labels

    models = []
    for nm in models_labels:
        parameters["model_name"] = nm
        loader, m = get_model_dataset(parameters, device=parameters.get("device","cuda"), return_loader=True)
        models.append(m)

    nb_batches = parameters.get("nb_batches",0)
    nb_inputs = parameters.get("nb_inputs", 0)
    i_inputs = 0

    direction = parameters.get("direction", "direct")

    for i_m, m in enumerate(models):
        print("model craft {}".format(i_m))

        attack = PGDL(m, eps=parameters.get("max_eps"), steps=parameters.get("max_iter"), random_start=False)
        attack.set_mode_targeted()

        parameters["model_craft"] = models_labels[i_m].split("-")[-1]
        experiment = init_comet(args=parameters, project_name=name)
        loss = SortAllClasses(criterion=parameters.get("criterion", None), experiment=experiment, direct=direction)

        for i, batch in enumerate(loader):
            if nb_batches > 0 and i >= nb_batches:
                break
            if nb_inputs > 0 and i_inputs >= nb_inputs:
                break

            imgs = batch['img'].to(attack.device)
            labels = batch['lab'].to(attack.device)
            labels[torch.isnan(labels)] = 0

            i_inputs += imgs.shape[0]

            with torch.no_grad():
                clean_output = m(imgs)
                target_outputs = get_less_cooccurred_target(clean_output, labels, map_target)


            advs= attack(imgs,target_outputs, loss)

            adv_output = m(advs)
            common_success_metrics(experiment, clean_output, adv_output, labels, target_outputs)


def build_cooccurences(names, document):
    from collections import OrderedDict
    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

    # Find the co-occurrences:
    for l in document:
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1

    return occurrences

def get_minimum_co_occurence(df,diseases, labels_to_index):
    class_mapping = {}
    available_labels = df.columns
    for output_labels, index in labels_to_index.items():
        if not output_labels in available_labels:
            continue

        filter =df[output_labels]
        min_cooccurence_index = filter[filter>0].argmin()
        min_cooccurence_label = filter[filter > 0].index[min_cooccurence_index]

        class_mapping[index] = labels_to_index[min_cooccurence_label]

    return class_mapping


def get_maximum_co_occurence(df, labels_to_index):
    class_mapping = {}
    available_labels = df.columns
    for output_labels, index in labels_to_index.items():
        if not output_labels in available_labels:
            continue

        filter = df[output_labels]
        cooccurence_index = filter[filter > 0].argmax()
        cooccurence_label = filter[filter > 0].index[cooccurence_index]

        class_mapping[index] = labels_to_index[cooccurence_label]

    return class_mapping


def csv_to_cooccurences(path):
    df = pd.read_csv(path)
    disease_df = df[df["Finding Labels"] != "No Finding"]
    diseases_list = disease_df["Finding Labels"].values
    diseases_combinations = [e.split("|") for e in diseases_list]
    unique_diseases = list(set([item for sublist in diseases_combinations for item in sublist]))

    cooccurences = build_cooccurences(unique_diseases, diseases_combinations)
    cooccurences_df = pd.DataFrame(cooccurences, columns=cooccurences.keys())

    #cooccurences_df.loc['Total',:] = cooccurences_df.sum(axis=0)

    return cooccurences_df, disease_df


def run(parameters, name="model_knowledge",model_labels=["densenet121-res224-all"]):
    print("running", parameters)

    if parameters.get("dataset") != "NIH":
        raise ValueError("only NIH correlation supported")

    path = os.path.join("data", 'NIH Chest X-rays', 'Data_Entry_2017.csv')
    cooccurences_df, disease_df = csv_to_cooccurences(path)
    loader, _ = get_model_dataset(parameters, device=parameters.get("device", "cuda"), return_loader=True)

    output_labels = loader.dataset.pathologies
    output_labels = {value: key for (key, value) in enumerate(output_labels)}

    # ap_target = get_minimum_co_occurence(cooccurences_df,disease_df, output_labels)

    columns = [output_labels[k] for k in cooccurences_df.columns]
    index = [output_labels[k] for k in cooccurences_df.index]
    cooccurences_df.index = index
    cooccurences_df.columns = columns
    vals = 1 - normalize(cooccurences_df.values, axis=0) - np.diagflat([1] * 14)
    cooccurences = pd.DataFrame(columns=columns, index=index, data=vals)
    # cooccurences = pd.DataFrame(columns=columns, index=index,
    #                            data=softmax(normalize(cooccurences_df.values, axis=0), 0))
    map_target = cooccurences[list(range(14))].reindex(list(range(14)))
    run_model(parameters, name=name, model_labels=model_labels, map_target=map_target.values)

if __name__ == '__main__':

    parameters = {"criterion": "sortedsampled", "algorithm": "pgd", "max_eps": 1/255, "workspace":"aaai22_health",
                  "norm": "Linf", "max_iter": 5, "eps_step": 0.001, "num_random_init": 1, "batch_size": 16,
                  "nb_batches":4,"lib": "torchattack", "model": "xrayvision", "dataset": "NIH", "reduction": "none",
                  "direction":"direct"}

    parameters["data_folder"] = os.path.join("data", 'NIH Chest X-rays')

    run(parameters)





