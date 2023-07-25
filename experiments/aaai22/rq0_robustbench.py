import sys
sys.path.append(".")

from comet_ml import Experiment
import torch
import random
from utils import init_comet

from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustbench.model_zoo import model_dicts as all_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

from utils.attacks import PGDL
from utils.metrics import topkaccuracy


def run(parameters, name="topk_accuracy", random_seed=0, experiment=None, topk=(1,3,5)):
    random.seed(random_seed)
    print("running", parameters)

    nb_batches = parameters.get("nb_batches", 0)
    batch_size = parameters.get("batch_size", 0)
    x_test, y_test = load_cifar10(n_examples=nb_batches*batch_size) if nb_batches is not None else load_cifar10()

    model_name = parameters.get("model")
    m = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')

    if experiment is None:
        experiment = init_comet(args= parameters, project_name=name)

    attack = PGDL(m, eps=parameters.get("max_eps"), steps=parameters.get("max_iter"), random_start=False)

    for i in range(len(x_test)//batch_size):
        if nb_batches is not None and nb_batches>0 and i>=nb_batches:
            break

        x = x_test[i*batch_size:(i+1)*batch_size]
        y = y_test[i*batch_size:(i+1)*batch_size]
        adv = attack(x,y, loss=parameters.get("criterion",None))

        with torch.no_grad():
            output = m(adv)

        acc = topkaccuracy(output,y.to(output.device),topk)
        experiment.log_metric("success_rate_top1",100-acc[0])
        if 3 in topk:
            experiment.log_metric("success_rate_top3", 100-acc[1])

        if 5 in topk:
            experiment.log_metric("success_rate_top5", 100-acc[2])



if __name__ == '__main__':
    parameters = {"use_hidden": False, "pretrained": True, "criterion": "ce", "algorithm": "PGD", "max_eps": 8/255,
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.05, "num_random_init": 1, "batch_size": 64, "nb_batches":None,
                  "lib": "torchattack", "model": "robustbench_Carmon2019Unlabeled", "dataset": "cifar10", "reduction": "none"}

    threat_model_ = ThreatModel("Linf")
    dataset_ = BenchmarkDataset("cifar10")
    models = list(all_models[dataset_][threat_model_].keys())

    epsilons = [8/255,16/255,0.1,0.3]
    algorithms = ["fgsm", "pgd", "apgd"]

    for algo in algorithms:
        for i, model in enumerate(models):
            for eps in epsilons:
                parameters["max_eps"] = eps
                parameters["model"] = model
                parameters["algorithm"] = algo

                topk =(1,3) if model=="retinopathy" else (1, 3, 5)

                run(parameters, name="topk_large", topk=topk)
