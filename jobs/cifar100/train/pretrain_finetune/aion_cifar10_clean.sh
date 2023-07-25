#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR100_R50_Pretrain"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=4:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.8.6-GCCcore-10.2.0

OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/c100/baseline_pretrain3"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"


python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "class_object#multilabel"  -name train-baseline-weight-finetune  --optimizer "sgd"  --lr 0.5  --dataset aux_cifar100_train --batch_size 512 --nb_secondary_labels 4 --model multi_task_resnet50 --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --num_epochs 200
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "class_object#multilabel"  -name train-baseline-weight-finetune --optimizer "sgd"  --lr 0.5  --dataset aux_cifar100_train --loss_jigsaw 1 --batch_size 512 --nb_secondary_labels 4 --permutations_jigsaw 4 --model multi_task_resnet50 --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --num_epochs 200
CUDA_VISIBLE_DEVICES=0 python experiments/train_baseline.py --labelfilter "class_object#multilabel"  -name train-baseline-weight-finetune --optimizer "sgd"  --lr 0.5  --dataset aux_cifar100_train --loss_rot 1 --batch_size 512 --nb_secondary_labels 4 --model multi_task_resnet50 --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --num_epochs 200