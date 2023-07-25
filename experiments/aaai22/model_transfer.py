import sys, os
sys.path.append(".")

from comet_ml import Experiment
import torch
import random
from utils import init_comet

from utils.losses import OrdinalLoss

from utils.models import get_model_dataset
from utils.attack import PGDL
from utils.metrics import topkaccuracy

def run(parameters, name="topk_accuracy", random_seed=0, model_labels = None ):
    print("running", parameters)

    models_labels = ["densenet121-res224-all", "resnet50","densenet121-res224-nih", "densenet121-res224-chex",
                     "densenet121-res224-pc", "densenet121-res224-nb", "densenet121-res224-ch",
                     "densenet121-res224-rsna"] if model_labels is None else model_labels

    models_labels = ["resnet50"]
    models = []
    for nm in models_labels:
        parameters["model_name"] = nm
        loader, m = get_model_dataset(parameters, device=parameters.get("device","cuda"), return_loader=True, normalized=False)
        models.append(m)


    nb_batches = parameters.get("nb_batches",0)

    for i_m, m in enumerate(models):
        #if i_m<1:
        #    continue
        max_eps = parameters.get("max_eps")
        eps = parameters.get("max_eps")
        max_iter = parameters.get("max_iter")
        print("model craft {}:{}".format(i_m, models_labels[i_m]))
        attack = PGDL(m, eps=max_eps, steps=max_iter, random_start=False)
        for i_t, m_test in enumerate(models):
            parameters["model_craft"] = models_labels[i_m].split("-")[-1]
            parameters["model_evaluate"] = models_labels[i_t].split("-")[-1]
            random.seed(random_seed)

            experiment = init_comet(args=parameters, project_name=name)
            (train_loader, loader), _ = get_model_dataset(parameters, device=parameters.get("device", "cuda"), return_loader=True,
                                          normalized=False, split=0.2)
            for i, batch in enumerate(loader):
                if nb_batches > 0 and i >= nb_batches:
                    print("max batches reached {}/{}".format(i,nb_batches))
                    break

                imgs = batch['img'].to(attack.device)
                labels = batch['lab'].to(attack.device)
                labels[torch.isnan(labels)] = 0
                if max_iter>0:
                    adv = attack(imgs, labels, loss=parameters.get("criterion", None))
                else:
                    adv = imgs
                print("batch {} model test {}".format(i,i_t))
                with torch.no_grad():
                    output = m_test(adv)
                    clean_output = m_test(imgs)
                    output_craft = m(imgs)

                    acc = topkaccuracy(output, clean_output, (1, 3, 5))
                    clean_acc = topkaccuracy(clean_output, labels, (1, 3, 5))
                    print(acc)
                    if name is not None:
                        experiment.log_parameter("nbclass_craft", output_craft.shape[1])
                        experiment.log_parameter("nbclass_test", clean_output.shape[1])
                        experiment.log_metric("top1_acc", acc[0])
                        experiment.log_metric("top3_acc", acc[1])
                        experiment.log_metric("top5_acc", acc[2])

                        experiment.log_metric("clean_top1_acc", clean_acc[0])
                        experiment.log_metric("clean_top3_acc", clean_acc[1])
                        experiment.log_metric("clean_top5_acc", clean_acc[2])


if __name__ == '__main__':

    parameters = {"criterion": "ord", "algorithm": "pgd", "max_eps": 1/255, "workspace":"aaai22_health",
                  "norm": "Linf", "max_iter": 0, "eps_step": 0.001, "num_random_init": 1, "batch_size": 16,
                  "nb_batches":4,"lib": "torchattack", "model": "xrayvision", "dataset": "NIH", "reduction": "none"}

    parameters["data_folder"] = os.path.join("data",'NIH Chest X-rays') if parameters.get("dataset")=="NIH" else ""
    run(parameters,  name="model_transfer")
