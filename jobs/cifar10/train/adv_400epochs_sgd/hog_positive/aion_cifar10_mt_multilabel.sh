#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_R50_hog1_Multilabel"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.8.6-GCCcore-10.2.0

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "class_object#multilabel" --num_epochs 400 --lr 0.1 --optimizer "sgd" --steps 10 --attack_target "multilabel"  -name train-baseline --dataset aux_cifar10_train --loss_hog 1 --threads 16 --batch_size 128 --model multi_task_resnet50 --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/baseline