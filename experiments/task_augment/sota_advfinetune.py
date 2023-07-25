import sys

sys.path.append(".")

import json, hashlib
from utils import init_comet

from utils.xrayvision import init_seed, init_tasks
from utils.train_utils import train
from experiments import get_argparser
from utils.models import get_model_dataset

parser = get_argparser("adv_finetune", train_arguments=True)
cfg = parser.parse_args()
print(cfg)


def run(parameters, model_labels = None ):
    print("running", parameters)

    models_labels = ["densenet121-res224-all", "resnet50","densenet121-res224-nih", "densenet121-res224-chex",
                     "densenet121-res224-pc", "chexpert","jfhealthcare"] if (model_labels is None or model_labels[0]=="") else model_labels

    models_labels = ["resnet50"]
    for nm in models_labels:
        parameters["model_name"] = nm
        dataset, model = get_model_dataset(parameters, split=True, return_loader=False,normalized=False)
        train_dataset, test_dataset = dataset
        experiment = init_comet(args=parameters, project_name=parameters["xp"])

        # fine-tune models
        train(model, train_dataset, cfg, experiment)


if __name__ == '__main__':

    cfg.tasks, _ = init_tasks(cfg, tasks_labels=cfg.labelfilter.split("-"))
    cfg.model = "xrayvision"
    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["xp"] = "sota_xray_train"
    parameters["workspace"] = "task-augment"


    # Setting the seed
    init_seed(cfg)

    model_labels = None
    run(parameters,  model_labels)
