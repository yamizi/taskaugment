import os, sys
sys.path.append(".")
from utils import init_comet
import json, hashlib
import numpy as np

from utils.xrayvision import init_seed, init_dataset, init_model
from utils.train_utils import train
from experiments import get_argparser

parser = get_argparser("xray",train_arguments=True)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':

    if os.path.exists("/mnt/lscratch/users/sghamizi/models/"):
        cfg.output_dir = cfg.output_dir.replace("users/yletraon","users/sghamizi")
        cfg.dataset_dir = cfg.dataset_dir.replace("users/yletraon", "users/sghamizi")

    elif os.path.exists("/mnt/lscratch/users/yletraon/models/"):
        cfg.output_dir = cfg.output_dir.replace("users/sghamizi","users/yletraon")
        cfg.dataset_dir = "/work/projects/medical-generalization/datasets"
        
    cfg.save_skip = 5
    cfg.main_metric = "auc"
    permutation = np.array([np.random.permutation(cfg.sections_jigsaw) for i in range(cfg.permutations_jigsaw-1)])
    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["algorithm"] = "PGD"

    if cfg.loss_jigsaw:
        parameters["permutation"] = cfg.permutation = permutation

    experiment = init_comet(args=parameters, project_name=cfg.name)

    # Setting the seed
    init_seed(cfg)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg, cfg.img_size)
    if cfg.random_labels:
        train_dataset.labels[:,-1] = np.random.permutation(train_dataset.labels[:,-1])
    # create models
    model, surrogates = init_model(cfg, train_dataset)

    train(model, train_dataset, cfg, experiment)
    print("Done")