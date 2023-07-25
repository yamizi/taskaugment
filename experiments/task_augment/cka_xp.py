import sys
sys.path.append(".")

from utils import init_comet

import torch
import json

from os.path import join
import argparse
from utils.xrayvision import init_dataset, init_model
from utils.cka import CKA
parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str,  default="cka", help='')
parser.add_argument('--weights_file', default="", type=str)
parser.add_argument('--weights_minsteps', type=int, default=1, help='')
parser.add_argument('--weights_maxsteps', type=int, default=100, help='')
parser.add_argument('--weights_step', type=int, default=5, help='')
parser.add_argument('--output_dir', type=str, default="D:/models")

parser.add_argument('--skip_layers', type=int, default=25, help='')
parser.add_argument('--skip_batch', type=int, default=3, help='')
parser.add_argument('--strategy', type=str, default="EPOCHS", help='')

parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="D:/datasets")
parser.add_argument('--model', type=str, default="multi_task_resnet50")
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--seed', type=int, default=0, help='')
parser.add_argument('--cuda', type=bool, default=True, help='')
parser.add_argument('--shuffle', type=bool, default=True, help='')
parser.add_argument('--threads', type=int, default=4, help='')
parser.add_argument('--labelfilter', type=str, default="Edema", help='')
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


if __name__ == '__main__':

    parameters = vars(cfg)
    dataset_name = parameters.get("dataset","chex")
    project_name = parameters.get("name","cka")
    labelfilter = parameters.get("labelfilter", "Edema")
    workspace_name = parameters.get("workspace","task-augment")

    model_dir = join(cfg.output_dir,dataset_name,labelfilter)
    model_dir2 = join(cfg.output_dir,cfg.weights_file)

    skip_layers = cfg.skip_layers

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model1 = init_model(cfg, test_dataset)
    # create models
    model2 = init_model(cfg, test_dataset)

    experiment = init_comet(args=parameters, project_name=project_name, workspace=workspace_name)

    data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads,
                                               pin_memory=cfg.cuda)

    for i in range(cfg.weights_minsteps,cfg.weights_maxsteps-1,cfg.weights_step):
        # loading model weights

        weights_file1 = join(model_dir, f"e{i}.pt")

        if cfg.strategy=="SELF":
            weights_file2 = weights_file1
        if cfg.strategy == "EPOCHS":
            weights_file2 = join(model_dir, f"e{i + 1}.pt")
        else:
            weights_file2 = join(model_dir2, f"e{i}.pt")
        try:
            model1.load_state_dict(torch.load(weights_file1).state_dict())
            model2.load_state_dict(torch.load(weights_file2).state_dict())
        except Exception as e:
            print("error loading models", weights_file1, weights_file2)
            continue

        layer_names_1 = [module_name for (module_name,module) in model1.named_modules() if not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
        layer_names_1 = [e for (i,e) in enumerate(layer_names_1) if i % skip_layers==0]

        layer_names_2 = [module_name for (module_name, module) in model2.named_modules() if
                         not isinstance(module, torch.nn.Sequential) and "conv" in module_name]
        layer_names_2 = [e for (i, e) in enumerate(layer_names_2) if i % skip_layers == 0]

        print(layer_names_1)

        with torch.no_grad():
            cka = CKA(model1, model2,
                      model1_name=weights_file1,
                      model2_name=weights_file2,
                      model1_layers=layer_names_1,
                      model2_layers=layer_names_2,
                      device="cuda" if cfg.cuda else"cpu")

            cka.compare(data_loader, max_batches=cfg.skip_batch)# if cfg.strategy=="SELF" else True)
            results = cka.export()
        results["CKA"] = results.get("CKA").cpu().numpy().tolist()
        experiment.log_asset_data(json.dumps(results), f"file{i}_{i+1}.json")
        #experiment.log_table(f"file{i}_{i+1}.csv",)