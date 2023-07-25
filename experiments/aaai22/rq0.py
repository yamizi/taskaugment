import sys
sys.path.append(".")

from comet_ml import Experiment
import torch
import random

from utils import init_comet
from utils.models import get_model_dataset
from utils.attacks import PGDL

from utils.metrics import topkaccuracy, success_forbid, success_sort_ab, success_force_all, success_sort_all


def run(parameters, name="topk_accuracy", random_seed=0, experiment=None, topk=(1,3,5)):
    random.seed(random_seed)
    print("running", parameters)
    loader, m = get_model_dataset(parameters, device=parameters.get("device","cuda"), return_loader=True)

    if experiment is None:
        experiment = init_comet(args= parameters, project_name=name)

    nb_batches = parameters.get("nb_batches",0)

    attack = PGDL(m, eps=parameters.get("max_eps"), steps=parameters.get("max_iter"), random_start=False)

    for i, batch in enumerate(loader):
        if nb_batches is not None and nb_batches>0 and i>=nb_batches:
            break

        adv = attack(batch[0],batch[1], loss=parameters.get("criterion",None))

        with torch.no_grad():
            output = m(adv)
            #clean_output = m(batch[0])

        topk_acc= topkaccuracy(output,batch[1].to(output.device),topk)
        forbid_acc = success_forbid(output,batch[1].to(output.device),topk)
        sort_ab_acc = success_sort_ab(output, torch.randint(output.shape[1],batch[1].shape,device=output.device),batch[1].to(output.device))
        force_all_acc = success_force_all(output, torch.randint(output.shape[1],(3,batch[1].shape[0]),device=output.device),topk)
        sort_all_acc = success_sort_all(output, torch.randint(output.shape[1],(3,batch[1].shape[0]),device=output.device))

        experiment.log_metric("success_rate_top1",100-topk_acc[0])
        experiment.log_metric("forbid_acc_top1", 100 - forbid_acc[0])
        if 3 in topk:
            experiment.log_metric("success_rate_top3", 100-topk_acc[1])
            experiment.log_metric("forbid_acc_top3", 100 - forbid_acc[1])

        if 5 in topk:
            experiment.log_metric("success_rate_top5", 100-topk_acc[2])
            experiment.log_metric("forbid_acc_top3", 100 - forbid_acc[2])

if __name__ == '__main__':
    parameters = {"use_hidden": False, "pretrained": True, "criterion": "ce", "algorithm": "PGD", "max_eps": 8/255,
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.05, "num_random_init": 1, "batch_size": 32,
                  "nb_batches":4, "lib": "torchattack", "model": "cifar10_resnet20", "dataset": "cifar10"}

    run(parameters, name="topk_accuracy")

    parameters["batch_size"]= 64
    parameters["nb_batches"]= None

    loss = ["ce", "focal", "ce"]
    models = ["imagenet_resnet50", "chest_xray","retinopathy"]
    datasets = ["imageNet","nih-xray", "aptos2019"]
    algorithms = ["fgsm", "pgd", "apgd"]
    epsilons = [4/255,8/255,16/255,0.1,0.3]


    for algo in algorithms:
        for i, model in enumerate(models):
            for eps in epsilons:
                parameters["max_eps"] = eps
                parameters["model"] = model
                parameters["criterion"] = loss[i]
                parameters["dataset"] = datasets[i]
                parameters["algorithm"] = algo

                topk =(1,3) if model=="retinopathy" else (1, 3, 5)

                run(parameters, name="topk_large", topk=topk)
