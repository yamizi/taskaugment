# GAT: Guided Adversarial Training with Pareto-optimal Auxiliary Tasks
Accepted at ICML 2023: https://proceedings.mlr.press/v202/ghamizi23a/ghamizi23a.pdf

## Abstract

While leveraging additional training data is well established to improve adversarial robustness, it incurs the unavoidable cost of data collection and the heavy computation to train models. To mitigate the costs, we propose \textit{Guided Adversarial Training } (GAT), a novel adversarial training technique that exploits auxiliary tasks under a limited set of training data. 
Our approach extends single-task models into multi-task models during the min-max optimization of adversarial training, and drives the loss optimization with a regularization of the gradient curvature across multiple tasks.
GAT leverages two types of auxiliary tasks: self-supervised tasks, where the labels are generated automatically, and domain-knowledge tasks, where human experts provide additional labels. Experimentally, under limited data, GAT increases the robust accuracy on CIFAR-10 up to four times (from 11\% to 42\% robust accuracy) and the robust AUC of CheXpert medical imaging dataset from 50\% to 83\%. On the full CIFAR-10 dataset, GAT outperforms eight state-of-the-art adversarial training strategies.

Our large study across five datasets and six tasks demonstrates that task augmentation is an efficient alternative to data augmentation, and can be key to achieving both clean and robust performances.


## Licence
MIT Licence



## External libraries & licences

```
./torchxrayvision
```
Adapted from [Torchxrayvision library](https://github.com/mlmed/torchxrayvision) Apache Licence


```
./utils/multitask_models
```
Adapted from [Taskonomy/Taskgrouping library](https://github.com/tstandley/taskgrouping/) MIT Licence


```
./utils/weights
```
Adapted from [LibMTL library](https://github.com/median-research-group/LibMTL) MIT Licence

## Install

Tested with Python 3.8 &  torch 1.10.0

```shell script
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

The external datasets are CheXpert and NIH; they are available from their original sources:

### CHEX dataset

* Dataset release website and download [HERE](https://stanfordmlgroup.github.io/competitions/chexpert/)

### NIH dataset

* Dataset [release website HERE](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
    
* Download full size images [HERE](https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a)

### CIFAR-10 unlabelled data augmentation 
* Self-supervised pseudo labelled data [HERE](https://drive.google.com/file/d/1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi/view)

## Packages


### Experiments shell scripts
```
./experiments/__init__.py
```
The arguments supported by the main experiments to train and evaluate the multi-task models


### Training the models
```
python experiments/train_baseline.py --labelfilter "multilabel" --attack_target "multilabel" --loss_rot 1 -name "" --dataset aux_cifar10_train --output_dir "./models" --dataset_dir "./datasets"
```

Will train a Resnet50 model using ATTA adversarial training on the main cifar10 task (multilabel) with a rotation auxiliary task and equal weighting strategy
* train_baseline.py is for traning cifar10, iagenet, stl, ... models
* train_xrayvision.py is for training chest xray models

### Testing the models
```
python experiments/test_model.py --labelfilter "multilabel" --attack_target "multilabel" --loss_rot 1 -name "" --dataset aux_cifar10_train --output_dir "./models" --dataset_dir "./datasets" --weights_file "my-model-weight.pt"
```

Will test a Resnet50 model with a rotation auxiliary task preloaded from **my-model-weight.pt**  using PGD attack on the main cifar10 task (multilabel) 
make sure to use the auxiliary task (--loss_rot 1; --loss_jigsaw 1) that matches the loaded file weight

### Experiments recording
This project uses [CometMl](www.comet.ml) to track the experiments and the results.
If you pass an experiment **name**, you need to create an account on comet and update the API key in **/app_config.py**

## Replication

### Pretrained models

You can find the pretrained models split in different archives [in this shared folder](https://figshare.com/projects/ATTA/139864)

The weight files provided are named with indication about which auxiliary task needs to be enabled.
* **multilabel-macro** in the filename means you need to use as script argument "--labelfilter 'multilabel-macro' "
* **rot-1.0** in the filename means you need to use as script argument "--loss_rot 1"
* **jigsaw-1.0** in the filename means you need to use as script argument "--loss_jigsaw 1"

You can have more details about how the models name are built in the method *train* from the file **utils/train_utils.py** 

### Experiments replication

We provide all the shell scripts used in the paper to **train** and evaluate the **perf**ormance of the models in folder */jobs* 
