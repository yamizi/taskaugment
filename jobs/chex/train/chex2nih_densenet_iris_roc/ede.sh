#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX2NIH_DENSENET_transfer_EDE"
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
ATTACK="Edema"
MODELS='Edema Cardiomegaly-Edema Pneumonia-Edema Consolidation-Edema Atelectasis-Edema Effusion-Edema Pneumothorax-Edema'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/transfer.py" -name transfer_roc_densenet --model multi_task_densenet --dataset $DATASET --weights_file "best/densenet/densenet-$MODEL.pt" --batch_size 64 --labelfilter $MODEL --attack_target "" --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$MODELDATASET"
count++
done