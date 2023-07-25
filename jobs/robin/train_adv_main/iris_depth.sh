#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "Robin_R50_depth_adv_main"
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
module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Python/3.8.6-GCCcore-10.2.0

cd ~/TopK/

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0
pip install --user  -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "multilabel"  --attack_target "multilabel" -workspace "dense" -name train-Robin-w_robin --dataset aux_robin_train --lr 0.1 --num_epochs 400  --optimizer "sgd"  --loss_depth 1 --threads 8 --batch_size 64 --model multi_task_resnet50 --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/baseline_weight_robin --img_size 224 --output_size 10 --pretrained "imagenet" --weight_strategy ${1:-ew} --force_cosine ${2:-0}