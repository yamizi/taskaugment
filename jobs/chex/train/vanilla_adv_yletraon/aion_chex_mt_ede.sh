#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_Resnet50_h_multitask_All"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=9:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.8.6-GCCcore-10.2.0

DATASETDIR=" /work/projects/medical-generalization/datasets"
SOURCEDIR="/mnt/irisgpfs/projects/multi_task_chest_xray"
OUTPUTDIR='/mnt/lscratch/users/yletraon/models/xray-resne50'

pip install --user  -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python experiments/yamizi/sota_advfinetune.py --lr 0.1 --max_eps 4 --step_eps 1 --optimizer "sgd" --steps 10 --num_epochs 50  -name "adv_finetune" --batch_size 180 --img_size 256 --main_metric "auc" --attack_target "Edema" --labelfilter "Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation-Pneumothorax-Pneumonia"  -name train-xray-resnet50 --dataset chex --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR