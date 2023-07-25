import os, sys
sys.path.append(".")

from utils import init_comet
import json, hashlib
import numpy as np

from utils.models import get_model_dataset
from utils.xrayvision import init_seed, init_tasks, init_dataset, init_model
from utils.test_utils import test
from experiments import get_argparser

parser = get_argparser("",train_arguments=False)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':

    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["xp"] = "pairwise"
    parameters["workspace"] = "task-augment"
    parameters["aux"] = "jigsaw" if "jigsaw" in cfg.weights_file else ("rotation" if "rot" in cfg.weights_file else ("macro" if "macro" in cfg.weights_file else "none"))

    surrogates = None
    weight_strategy = cfg.weight_strategy
    parameters["weight_strategy"] = cfg.weights_file.split("w-")[1][0:-3] if len(cfg.weights_file.split("w-"))>1 else weight_strategy

    if cfg.loss_jigsaw:
        permutation = np.array([np.random.permutation(cfg.sections_jigsaw) for i in range(cfg.permutations_jigsaw - 1)])
        parameters["permutation"] = cfg.permutation = permutation

    experiment = init_comet(args=parameters, project_name=cfg.name)

    # Setting the seed
    init_seed(cfg)

    if "cifar10" in cfg.dataset:
        parameters["output_size"] = 10
        parameters["img_size"] = 32
        parameters["train"] = False
        parameters["tasks"], _, surrogates = init_tasks(cfg, tasks_labels=cfg.labelfilter.split("-"))
        test_dataset, model = get_model_dataset(parameters, parameters.get("shuffle"),
                                           "cuda" if parameters.get("cuda") else "cpu",
                                           return_loader=False)
        test_dataset.set_includes(cfg)

    else:
        parameters["output_size"] = 1
        parameters["img_size"] = 512

        # Setting the dataset
        train_dataset, test_dataset = init_dataset(cfg, cfg.img_size)

        # create models
        model,surrogates = init_model(cfg, test_dataset)


    # test models
    test(model, test_dataset, cfg, experiment, limit=cfg.batch_limit, surrogates=surrogates)
