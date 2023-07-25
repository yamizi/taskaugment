import sys
sys.path.append(".")
from utils import init_comet
import torch
import numpy as np
import json, hashlib
from os.path import join
from utils.xrayvision import init_seed, init_dataset, init_model
from experiments import get_argparser
from utils.train_utils import train

parser = get_argparser("finetune",train_arguments=True)
parser.add_argument('--labelsource', type=str, default="Atelectasis", help='')
parser.add_argument('--source_dir', type=str, default="D:/models", help='')
parser.add_argument('--strategy', type=str, default="FINETUNE", help='')

#strategy = FINETUNE / EXTRACTOR
cfg = parser.parse_args()
print(cfg)


if __name__ == '__main__':
    init_seed(cfg)
    cfg.dataset="chex"

    if cfg.loss_rot!=0:
        cfg.data_aug_rot = 0

    permutation = np.array([np.random.permutation(cfg.sections_jigsaw) for i in range(cfg.permutations_jigsaw-1)])
    parameters = vars(cfg)
    m = hashlib.md5(json.dumps(parameters, sort_keys=True).encode('utf-8'))
    uniqueid = m.hexdigest()
    parameters["uniqueid"] = uniqueid
    parameters["algorithm"] = "PGD"
    parameters["max_eps"] = 0

    if cfg.loss_jigsaw:
        parameters["permutation"] = cfg.permutation = permutation

    experiment = init_comet(args=vars(cfg), project_name=cfg.name, workspace=cfg.workspace)

    weights_file = join(cfg.source_dir,cfg.dataset,"best", f"{cfg.labelsource}.pt")
    print(f"loading model to from {weights_file} to finetune to {cfg.labelfilter}")
    initial_weights= torch.load(weights_file).state_dict()

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model1 = init_model(cfg, test_dataset)


    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)

    initial_weights = dict(zip(model1.state_dict().keys(),initial_weights.values()))
    model1.load_state_dict(initial_weights)

    if cfg.strategy=="EXTRACTOR":
        ## Freeze all layers but decoder
        nb_params = len(list(model1.parameters()))
        for i, param in enumerate(model1.parameters()):
            if i < nb_params-2:
                param.requires_grad = False

    train(model1, train_dataset, cfg, experiment)
