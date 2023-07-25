This repository is about Robust Multitask Learning and has multiple branches:

# GAT: Guided Adversarial Training with Pareto-optimal Auxiliary Tasks
Accepted at ICML 2023: https://proceedings.mlr.press/v202/ghamizi23a/ghamizi23a.pdf
For our GAT paper (ICML23), checkout the branch icml23

# Adversarial robustness in multi-task learning: Promises and illusions
Accepted at AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/19950/19709
For our WGD paper (AAAI22), checkout the main 


# Adversarial Robustness in Multi-Task Learning:Promises and Illusions, by Salah Ghamizi, Maxime Cordy, Mike Papadakis, and Yves Le Traon

This is the source code for the paper "Adversarial Robustness in Multi-Task Learning:Promises and Illusions" accepted at AAAI2022.
[Preprint PDF](https://arxiv.org/pdf/2110.15053)

Our experiments use CometML to track and record the results of our experiments. You will need a valid (free) account from [here](https://www.comet.ml/) and a personal API Key to send the results of the experiments.

In the replication package you will find three folders:

* MTRobust: cloned from (https://github.com/columbia/MTRobust/) and extended with additional models (Xception, WideResnet). This repository is used to train the models following the same setting as the original paper of MTRobust. Read their documentation for more instructions about how to train models. Or use our scripts in MTRobust/jobs/

You will need to download the Taskonomy dataset as explained in (https://github.com/columbia/MTRobust/) and update the configuration files in MTRobust/jobs/

* MTVulnerability: Our package to attack and evaluate the vulnerability of the models. The folder MTVulnerability/jobs contains the script you can run directly. Please read the specific README file of our package for more details & instructions.

* models: This folder provides one pretrained taskonomy model for task combination s (semantic segmentation) and d (Z-depth) to use as a quick test. You can use this folder for the variable **MODEL** in the scripts located in *MTVulnerability/jobs*
