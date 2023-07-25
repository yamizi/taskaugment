import sys

sys.path.append(".")

from comet_ml import Experiment
from utils import init_comet

import torch
from glob import glob
from os.path import exists, join
import os
import shutil
import json
import numpy.ma as ma
from torch.nn import CosineSimilarity
from scipy import spatial
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-name', type=str)
parser.add_argument('-workspace', type=str)
parser.add_argument('-dataset', type=str, default="chex", help='')
parser.add_argument('--input_dir', default="/mnt/irisgpfs/projects/multi_task_chest_xray", type=str)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':

    parameters = vars(cfg)
    dataset_name = parameters.get("dataset","chex")
    input_dir = parameters.get("input_dir","/mnt/irisgpfs/projects/multi_task_chest_xray")
    project_name = parameters.get("name","gradients")
    workspace_name = parameters.get("workspace","task-augment")

    input_dir = join(input_dir,dataset_name,"gradients")
    #input_dir = "C:/Users/salah.ghamizi/Documents/PhD/Code/TopK/data"

    folder = join(input_dir, '{}*-gradients-b*.pkl'.format(dataset_name))
    gradients_files = glob(folder)
    print("gradient files", len(gradients_files),folder)
    df = pd.DataFrame(columns=["id","dataset","batch","attack_step","masked_covariance","masked_cosine","covariance","cosine"])

    parameters = {"xp": id, "dataset": dataset_name}
    experiment = init_comet(args=parameters, project_name=project_name, workspace=workspace_name)


    for gradient_file in gradients_files:

        id = gradient_file.split("task_resnet50-")[1].split("-gradients")[0]
        batch_id = gradient_file.split("-gradients-b")[1][:-4]

        grads = torch.load(gradient_file)

        for (i,attack_step) in enumerate(grads):
            gradients_0 = list(attack_step.values())
            if len(gradients_0)>1:
                task_1 = gradients_0[0].flatten()
                task_2 = gradients_0[1].flatten()

                cos = CosineSimilarity(dim=0, eps=1e-6)
                masked_covariance = ma.cov(ma.masked_invalid(task_1.numpy()),ma.masked_invalid(task_2.numpy())).tolist()
                masked_cosine = spatial.distance.cosine(ma.masked_invalid(task_1.numpy()).filled(0),
                                        ma.masked_invalid(task_2.numpy()).filled(0))

                cosine = cos(task_1, task_2).numpy().tolist()
                covariance = []#torch.cov(torch.cat([task_1.unsqueeze(0),task_2.unsqueeze(0)])).numpy().tolist()

                row = {"xp":id, "dataset":dataset_name,"batch":batch_id,"masked_covariance":masked_covariance,
                           "masked_cosine":masked_cosine,"covariance":covariance,"cosine":cosine, "attack_step":i}
                df = df.append(row,ignore_index=True)
                #experiment.log_asset_data(json.dumps(row),"similarity")

                print(covariance,masked_covariance,cosine,masked_cosine)

    experiment.log_dataframe_profile(df,name="gradients")

