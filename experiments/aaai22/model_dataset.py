import sys, os

sys.path.append(".")

from comet_ml import Experiment
import torch


from experiments.aaai22.model_transfer import run

if __name__ == '__main__':

    parameters = {"criterion": "bce", "algorithm": "pgd", "max_eps": 1/255, "workspace":"aaai22_health",
                  "norm": "Linf", "max_iter": 100, "eps_step": 0.001, "num_random_init": 1, "batch_size": 16,
                  "nb_batches":4,"lib": "torchattack", "model": "xrayvision", "dataset": "NIH", "reduction": "none",
                  "data_folder":os.path.join("data", 'NIH Chest X-rays')}

    models_labels = ["densenet121-res224-all", "resnet50", "densenet121-res224-nih", "densenet121-res224-chex"]

    datasets = [{"dataset":"NIH","data_folder":os.path.join("data", 'NIH Chest X-rays')},
                {"dataset":"PC","data_folder":{"csv":"D://datasets//PC//PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv","img":"D://datasets//PC//images-224//images-224"}},
                {"dataset":"CHEX","data_folder":{"csv":"D://datasets//CheXpert//train.csv","img":"D://datasets//CheXpert"}}]

    for dt in datasets:
        parameters = {**parameters,**dt}
        run(parameters,  name="model_dataset", model_labels=models_labels)
