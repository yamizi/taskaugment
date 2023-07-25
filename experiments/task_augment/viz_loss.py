import os.path
import sys, random
sys.path.append(".")
sys.path.append("./GradVis-master/toolbox")

from utils import init_comet

import torch
import numpy as np

from os.path import join
import argparse
from utils.xrayvision import init_dataset, init_model
import Visualization as vis
import nn_model
import trajectory_plots as tplot

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str,  default="viz-learning-loss", help='')
parser.add_argument('-workspace', type=str,  default="task-augment", help='')
parser.add_argument('--weights_file', default="", type=str)
parser.add_argument('--weights_step', type=int, default=10, help='')
parser.add_argument('--output_dir', type=str, default="D:/models")
parser.add_argument('--weights_minsteps', type=int, default=1, help='')
parser.add_argument('--weights_maxsteps', type=int, default=100, help='')

parser.add_argument('--dataset', type=str, default="chex")
parser.add_argument('--dataset_dir', type=str, default="D:/datasets")
parser.add_argument('--model', type=str, default="multi_task_resnet50")
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--batch_max', type=int, default=0, help='')
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
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset_name = cfg.dataset
    project_name = cfg.name
    label_filter = cfg.labelfilter
    workspace_name = cfg.workspace

    model_dir = join(cfg.output_dir,dataset_name,label_filter)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model = init_model(cfg, test_dataset)

    experiment = init_comet(args= vars(cfg), project_name=project_name, workspace=workspace_name)

    data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=cfg.shuffle,
                                               num_workers=cfg.threads,
                                               pin_memory=cfg.cuda)

    #weights_file = join(model_dir, "e*.pt")
    #checkpoint_files = glob(weights_file)
    num_files = (cfg.weights_maxsteps-cfg.weights_minsteps + 1)
    fileindices = np.linspace(cfg.weights_minsteps, cfg.weights_maxsteps,num_files, dtype=int)

    checkpoints = [join(model_dir, f"e{f}.pt") for f in fileindices if os.path.exists(join(model_dir, f"e{f}.pt"))]
    checkpoint_files = [f for (i,f) in enumerate(checkpoints) if i%cfg.weights_step==0]
    print(checkpoint_files)
    nnmodel = nn_model.PyTorch_NNModel(model, cfg, checkpoint_files[-1],data_loader=data_loader, max_batch=cfg.batch_max)
    vis.visualize(nnmodel, checkpoint_files, 50, "minima_vis_pca", proz=.4, verbose=True)
    tplot.plot_loss_2D("minima_vis_pca.npz", filename=join(model_dir,"resnet_minima_2D_plot_pca"))
    tplot.plot_loss_3D("minima_vis_pca.npz", filename=join(model_dir,"resnet_minima_3D_plot_pca"), degrees=50)

        #experiment.log_asset_data(json.dumps(results), f"file{i}_{i+1}.json")
        #experiment.log_table(f"file{i}_{i+1}.csv",)