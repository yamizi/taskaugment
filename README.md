This is the source code for the paper "Adversarial Robustness in Multi-Task Learning:Promises and Illusions"
Accepted at AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/19950/19709

Our experiments use CometML to track and record the results of our experiments. You will need a valid (free) account from https://www.comet.ml/ and a personal API Key to send the results of the experiments.

In the replication package you will find three folders:

* MTRobust: cloned from https://github.com/columbia/MTRobust/ and extended with additional models (Xception, WideResnet). This repository is used to train the models following the same setting as the original paper of MTRobust. Read their documentation for more instructions about how to train models. Or use our scripts in MTRobust/jobs/

You will need to download the Taskonomy dataset as explained in https://github.com/columbia/MTRobust/ and update the configuration files in MTRobust/jobs/

* MTVulnerability: Our package to attack and evaluate the vulnerability of the models. The folder MTVulnerability/jobs contains the script you can run directly. Please read the specific README file of our package for more details & instructions.

* models: This folder provides one pretrained taskonomy model for task combination s (semantic segmentation) and d (Z-depth) to use as a quick test. You can use this folder for the variable **MODEL** in the scripts located in *MTVulnerability/jobs*
