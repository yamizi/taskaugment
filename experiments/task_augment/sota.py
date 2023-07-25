import sys

sys.path.append(".")

import json, hashlib
import argparse
from utils import init_comet

from utils.xrayvision import init_seed, init_dataset
from utils.test_utils import test

from utils.models import get_model_dataset


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, default="", help='')
parser.add_argument('-name', type=str)
parser.add_argument('--weights_file', type=str)
parser.add_argument('--output_dir', type=str, default="D:/models")
#parser.add_argument('--dataset_split', type=str, default="test")
parser.add_argument('--dataset', type=str, default="nih")
parser.add_argument('--dataset_dir', type=str, default="D:/datasets")
parser.add_argument('--model', type=str, default="xrayvision")
parser.add_argument('--batch_size', type=int, default=16, help='')
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
parser.add_argument('--max_eps', type=int, default=4, help='')
parser.add_argument('--step_eps', type=int, default=1, help='')
parser.add_argument('--steps', type=int, default=4, help='')
parser.add_argument('--record_all_tasks', type=int, default=0, help='')
parser.add_argument('--loss_hog', type=float, default=0, help='')
parser.add_argument('--loss_ae', type=float, default=0, help='')
parser.add_argument('--loss_jigsaw', type=float, default=0, help='')
parser.add_argument('--loss_rot', type=float, default=0, help='')
parser.add_argument('--sections_jigsaw', type=int, default=16, help='')
parser.add_argument('--nb_secondary_labels', type=int, default=10, help='')

cfg = parser.parse_args()
print(cfg)


def run(parameters, test_dataset, model_labels = None ):
    print("running", parameters)

    models_labels = ["densenet121-res224-all", "resnet50","densenet121-res224-nih", "densenet121-res224-chex",
                     "densenet121-res224-pc", "chexpert","jfhealthcare"] if (model_labels is None or model_labels[0]=="") else model_labels


    models_labels = ["chexpert"]
    for nm in models_labels:
        parameters["model_name"] = nm
        dataset, model = get_model_dataset(parameters, split=True, return_loader=False,normalized=False)
        train_dataset, test_dataset = dataset
        experiment = init_comet(args=parameters, project_name=parameters["xp"])

        # test models
        test(model, test_dataset, cfg, experiment, load_weights=False)


if __name__ == '__main__':


    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["xp"] = "sota"
    parameters["workspace"] = "task-augment"

    # Setting the seed
    init_seed(cfg)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    model_labels = None
    run(parameters, test_dataset, model_labels)
