import os, sys
sys.path.append(".")
from utils import init_comet
import json, hashlib
import numpy as np

from utils.models import get_model_dataset
from utils.xrayvision import init_seed, init_tasks
from utils.train_utils import train
from experiments import get_argparser

parser = get_argparser("",train_arguments=True)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':

    if not "macro" in cfg.labelfilter:
        cfg.nb_secondary_labels = cfg.nb_rotations

    if cfg.attack_target=="":
        cfg.algorithm = ""

    if os.path.exists("/mnt/lscratch/users/sghamizi/models/"):
        cfg.output_dir = cfg.output_dir.replace("users/yletraon","users/sghamizi")
        cfg.dataset_dir = cfg.dataset_dir.replace("users/yletraon", "users/sghamizi")

    elif os.path.exists("/mnt/lscratch/users/yletraon/models/"):
        cfg.output_dir = cfg.output_dir.replace("users/sghamizi","users/yletraon")
        cfg.dataset_dir = cfg.dataset_dir.replace("users/sghamizi", "users/yletraon")

    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid

    parameters["output_size"] = 10 if "_cifar10_" in parameters.get("dataset") else 100 if "_cifar100_" in parameters.get("dataset") else parameters["output_size"]

    if cfg.loss_jigsaw:
        permutation = np.array([np.random.permutation(cfg.sections_jigsaw) for i in range(cfg.permutations_jigsaw - 1)])
        parameters["permutation"] = cfg.permutation = permutation


    experiment = init_comet(args=parameters, project_name=cfg.name)

    if experiment is not None:
        experiment.log_code(file_name="exploration/xrayvision/train_utils.py")
        experiment.log_code(folder="utils")

    # Setting the seed
    init_seed(cfg)

    # Setting the dataset & the models
    parameters["tasks"], _, surrogates = init_tasks(cfg, tasks_labels=cfg.labelfilter.split("-"))
    dataset, model = get_model_dataset(parameters,parameters.get("shuffle"),"cuda" if parameters.get("cuda") else "cpu",
                                       return_loader=False)
    dataset.set_includes(cfg)

    if cfg.random_labels:
        dataset.labels = np.random.permutation(dataset.labels)


    train(model, dataset, cfg, experiment, load_weights=cfg.weights_file!="", surrogates=surrogates)
    print("Done")