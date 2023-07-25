import sys
sys.path.append(".")
from utils import init_comet
import json, hashlib
import argparse


from utils.xrayvision import init_seed, init_dataset, init_model
from utils.test_utils import test

from experiments import get_argparser

parser = get_argparser("xray",train_arguments=False)
cfg = parser.parse_args()
print(cfg)

if __name__ == '__main__':

    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["xp"] = "transfer"
    parameters["workspace"] = "task-augment"
    parameters["model_dataset"] =parameters["output_dir"].split("/")[-1]

    experiment = init_comet(args=parameters, project_name=cfg.name)

    # Setting the seed
    init_seed(cfg)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model = init_model(cfg, test_dataset)


    # test models
    test(model, test_dataset, cfg, experiment)
