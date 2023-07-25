#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "C10_R50_Pre_adv_vanilla"
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

OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/madry_pretrain2"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
WEIGHTS="/mnt/lscratch/users/sghamizi/models/madry_pretrain/aux_cifar10/best/resnet50/madry-finetune-multilabel-macro_optim-sgd_cosine-0_w-ew.pt"
DATASUBSET=0.1

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "multilabel-macro" -name train-baseline-madry-partial  --attack_target "multilabel" --weight_strategy ${1:-mgda} --force_cosine ${2:-0} --algorithm ${3:-MADRY}  --data_subset $DATASUBSET --optimizer "sgd"  --lr 0.1  --weight_strategy ${1:-mgda} --force_cosine ${2:-0} --algorithm ${3:-MADRY}  --dataset aux_cifar10_train --batch_size 384 --nb_secondary_labels 2 --model multi_task_resnet50 --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --num_epochs 400 --seed 10