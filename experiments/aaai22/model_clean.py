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

    models = []
    for nm in models_labels:
        parameters["model_name"] = nm
        loader, m = get_model_dataset(parameters, device=parameters.get("device","cuda"), return_loader=True)
        models.append(m)


    nb_batches = parameters.get("nb_batches",0)

    for i_m, m in enumerate(models):
        #if i_m<1:
        #    continue
        print("model craft {}:{}".format(i_m, models_labels[i_m]))
        attack = PGDL(m, eps=parameters.get("max_eps"), steps=parameters.get("max_iter"), random_start=False)
        for i_t, m_test in enumerate(models):
            parameters["model_craft"] = models_labels[i_m].split("-")[-1]
            parameters["model_evaluate"] = models_labels[i_t].split("-")[-1]
            random.seed(random_seed)
            experiment = init_comet(args=parameters, project_name=name)
            loader, _ = get_model_dataset(parameters, device=parameters.get("device", "cuda"), return_loader=True)
            for i, batch in enumerate(loader):
                if nb_batches > 0 and i >= nb_batches:
                    print("max batches reached {}/{}".format(i,nb_batches))
                    break

                imgs = batch['img'].to(attack.device)
                labels = batch['lab'].to(attack.device)
                labels[torch.isnan(labels)] = 0
                with torch.no_grad():
                    clean_output = m_test(imgs)
                    experiment.log_asset_data(clean_output,"clean_output",step=i)
                    experiment.log_asset_data(labels, "labels", step=i)


if __name__ == '__main__':

    parameters = {"criterion": "bce", "algorithm": "pgd", "max_eps": 1 / 255, "workspace": "aaai22_health",
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.001, "num_random_init": 1, "batch_size": 16,
                  "nb_batches": 4, "lib": "torchattack", "model": "xrayvision", "dataset": "NIH", "reduction": "none",
                  "data_folder": os.path.join("data", 'NIH Chest X-rays')}

    models_labels = ["densenet121-res224-all", "resnet50", "densenet121-res224-nih", "densenet121-res224-chex"]

    datasets = [{"dataset": "PC",
                 "data_folder": {"csv": "D://datasets//PC//PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                                 "img": "D://datasets//PC//images-224//images-224"}},
                {"dataset": "NIH", "data_folder": os.path.join("data", 'NIH Chest X-rays')},
                {"dataset": "CHEX",
                 "data_folder": {"csv": "D://datasets//CheXpert//train.csv", "img": "D://datasets//CheXpert"}}]

    for dt in datasets:
        parameters = {**parameters, **dt}
        run(parameters, name="model_dataset", model_labels=models_labels)
