#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "NIH_CHEX_R50_transfer_ATE"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=2:00:00
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

DATASET="nih"
MODELDATASET="chex"
ATTACK="Atelectasis"
MODELS='Atelectasis Atelectasis-Cardiomegaly Atelectasis-Pneumonia Atelectasis-Effusion Atelectasis-Consolidation Atelectasis-Edema Atelectasis-Pneumothorax'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/transfer.py" -name transfer_large --dataset $DATASET  --loss_rot 1 --weights_file "best/${MODEL}_rot-1.0.pt" --batch_size 64 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$MODELDATASET"
count++
done


MODELS='Cardiomegaly Pneumonia Effusion Consolidation Edema Pneumothorax'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/transfer.py" -name transfer_large --dataset $DATASET  --loss_rot 1 --weights_file "best/${MODEL}_rot-1.0.pt" --batch_size 64 --labelfilter $MODEL --attack_target $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$MODELDATASET"
count++
done