import sys
sys.path.append(".")
from utils import init_comet
import json, hashlib
import argparse
import numpy as np

from utils.models import get_model_dataset
from utils.xrayvision import init_seed, init_tasks, init_dataset, init_model
from utils.test_utils import test


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str, default="local")
parser.add_argument('--weights_file', type=str)
parser.add_argument('--output_dir', type=str, default="D:/models")
parser.add_argument('--dataset', type=str, default="pc")
parser.add_argument('--dataset_dir', type=str, default="D:/datasets")
parser.add_argument('--model', type=str, default="multi_task_resnet50")
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--batch_limit', type=int, default=0, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--labelfilter', type=str, default="", help='')
parser.add_argument('--attack_target', type=str, default='', help='')
parser.add_argument('--data_aug', type=bool, default=True, help='')
parser.add_argument('--data_aug_rot', type=int, default=45, help='')
parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
parser.add_argument('--label_concat', type=bool, default=False, help='')
parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
parser.add_argument('--labelunion', type=bool, default=False, help='')
parser.add_argument('--loss_sex', type=float, default=0, help='')
parser.add_argument('--loss_age', type=float, default=0, help='')
parser.add_argument('--max_eps', type=int, default=8, help='')
parser.add_argument('--step_eps', type=int, default=2, help='')
parser.add_argument('--steps', type=int, default=4, help='')
parser.add_argument('--record_all_tasks', type=int, default=0, help='')
parser.add_argument('--record_roc', type=int, default=1, help='')
parser.add_argument('--loss_hog', type=float, default=0, help='')
parser.add_argument('--loss_ae', type=float, default=0, help='')
parser.add_argument('--loss_jigsaw', type=float, default=0, help='')
parser.add_argument('--loss_rot', type=float, default=0, help='')
parser.add_argument('--sections_jigsaw', type=int, default=16, help='')
parser.add_argument('--permutations_jigsaw', type=int, default=31, help='')
parser.add_argument('--nb_secondary_labels', type=int, default=10, help='')
parser.add_argument('--img_size', type=int, default=256, help='')

parser.add_argument('--labelfilter_surrogate', type=str, default="", help='')
parser.add_argument('--weights_file_surrogate', type=str)
cfg = parser.parse_args()
print(cfg)


def init_model(cfg, labelfilter):
    if "cifar10" in cfg.dataset:
        parameters["output_size"] = 10
        #parameters["img_size"] = 32
        parameters["train"] = False
        parameters["tasks"], _ = init_tasks(cfg, tasks_labels=labelfilter)
        test_dataset, model = get_model_dataset(parameters, parameters.get("shuffle"),
                                           "cuda" if parameters.get("cuda") else "cpu",
                                           return_loader=False)
        test_dataset.set_includes(cfg)

    else:
        # Setting the dataset
        train_dataset, test_dataset = init_dataset(cfg)

        # create models
        model = init_model(cfg, test_dataset)

    return model, test_dataset

if __name__ == '__main__':

    #if not hasattr(cfg,"weights_file") or cfg.weights_file is None or cfg.weights_file=="":
    #    cfg.weights_file="best/{}.pt".format(cfg.labelfilter)

    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["xp"] = "adv_transfer"
    parameters["workspace"] = "task-augment"

    if cfg.loss_jigsaw:
        permutation = np.array([np.random.permutation(cfg.sections_jigsaw) for i in range(cfg.permutations_jigsaw - 1)])
        parameters["permutation"] = cfg.permutation = permutation

    experiment = init_comet(args=parameters, project_name=cfg.name)

    # Setting the seed
    init_seed(cfg)

    model, test_dataset = init_model(cfg, cfg.labelfilter.split("-"))
    model_surrogate, _ = init_model(cfg, cfg.labelfilter_surrogate.split("-"))


    # test models
    test(model, test_dataset, cfg, experiment, limit=cfg.batch_limit, model_surrogate=model_surrogate)
