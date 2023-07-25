import sys
sys.path.append(".")
sys.path.append("./loss_landscapes_master")
from utils import init_comet
import torch
import numpy as np
from os.path import join
import abc
from utils.xrayvision import init_seed, init_dataset, init_model
from experiments import get_argparser
from tqdm import tqdm
import sklearn
from utils.test_utils import build_output, ml_to_mt
from loss_landscapes_master.loss_landscapes.model_interface.model_wrapper import ModelWrapper
from loss_landscapes_master.loss_landscapes.main import linear_interpolation
#from utils.models import compare_models

parser = get_argparser("interpolation")
parser.add_argument('--labelsource', type=str, default="Atelectasis", help='')
parser.add_argument('--labeltarget', type=str, default="Atelectasis", help='')
parser.add_argument('--interpolation_steps', type=int, default=50, help='')
parser.add_argument('--source_dir', type=str, default="D:/models", help='')
parser.add_argument('--target_dir', type=str, default="D:/models", help='')
parser.add_argument('--source_dataset', type=str, default="chex", help='')
parser.add_argument('--target_dataset', type=str, default="chex", help='')
#strategy = FINETUNE / EXTRACTOR
cfg = parser.parse_args()
print(cfg)

class Metric(abc.ABC):
    """ A quantity that can be computed given a model or an agent. """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __call__(self, model_wrapper: ModelWrapper):
        pass

class Loss(Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, dataloader, loss_fn=None, batch_max=0, device="cuda", target_auc=None, experiment=None):
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.BCEWithLogitsLoss()
        self.dataloader = dataloader
        self.max_batch = batch_max
        self.device = device
        self.target_auc = target_auc
        self.experiment = experiment

    def __call__(self, model_wrapper: ModelWrapper, step:int) -> float:

        pathologies = [f"class_object#{t}" for t in self.dataloader.dataset.pathologies]
        avg_loss = []
        task_outputs = {}
        task_targets = {}
        task_aucs = {}

        for i, task in enumerate(pathologies):
            task_outputs[task] = np.array([])
            task_targets[task] = np.array([])
            task_aucs[task] = []

        t = tqdm(self.dataloader)
        criterion = self.loss_fn

        for batch_idx, samples in enumerate(t):

            if self.max_batch > 0 and batch_idx >= self.max_batch:
                break

            images = samples["img"].to(self.device)
            targets = samples["lab"].to(self.device)
            if not isinstance(targets, dict):
                targets = ml_to_mt(targets, pathologies)

            if self.target_auc:
                targets = {k:v for (k,v) in targets.items() if self.target_auc in k}
            task_targets, task_outputs, avg_loss = build_output(model_wrapper, self.device, t, images, targets,
                                                                criterion,     task_outputs,
                                                                task_targets, avg_loss)

            for (task, batch_target) in task_targets.items():
                if len(np.unique(batch_target)) > 1:
                    task_auc = sklearn.metrics.roc_auc_score(batch_target, task_outputs[task])
                    task_aucs[task].append(task_auc)
                    if self.experiment is not None:
                        self.experiment.log_metric("{}_auc".format(task),task_auc, step=step, epoch=batch_idx)

            if self.experiment is not None:
                self.experiment.log_metric("loss", avg_loss[-1], step=step, epoch=batch_idx)

        return np.mean(avg_loss) if self.target_auc is None else np.mean(task_aucs["class_object#{}".format(self.target_auc)])

        #return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


if __name__ == '__main__':
    init_seed(cfg)
    cfg.dataset="chex"

    weights_file_source = join(cfg.source_dir, cfg.source_dataset, "best", f"{cfg.labelsource}.pt")
    weights_source = torch.load(weights_file_source).state_dict()
    weights_file_target = join(cfg.target_dir, cfg.target_dataset, "best", f"{cfg.labeltarget}.pt")
    weights_target = torch.load(weights_file_target).state_dict()
    print(f"interpolate model to from {weights_file_source} to {weights_file_target} evaluated on {cfg.labelfilter}")


    if weights_file_source==weights_file_target:
        exit()

    experiment = init_comet(args=vars(cfg), project_name=cfg.name, workspace=cfg.workspace)

    # Setting the dataset
    train_dataset, test_dataset = init_dataset(cfg)

    # create models
    model1 = init_model(cfg, test_dataset, cfg.labelsource.split("-"))
    model2 = init_model(cfg, test_dataset, cfg.labeltarget.split("-"))

    model1.load_state_dict(weights_source)
    model2.load_state_dict(weights_target)

    data_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.threads,
                                              pin_memory=cfg.cuda)

    # compute loss data
    metric = Loss(data_loader,device="cuda" if cfg.cuda else "cpu", batch_max=cfg.batch_max, target_auc=cfg.labelfilter, experiment=experiment)
    loss_data = linear_interpolation(model1, model2, metric, cfg.interpolation_steps, deepcopy_model=True)
    print(loss_data)

