#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX2NIH_FINE_R50_transfer_EFF"
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

MODELS='Atelectasis Cardiomegaly Pneumonia Effusion Consolidation Edema Pneumothorax'
MODEL="Effusion"

count=0
for MODELSOURCE in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/transfer.py" -name transfer_extract_roc --dataset $DATASET --weights_file "best/${MODELSOURCE}_src-${MODEL}_str-EXTRACTOR.pt" --batch_size 256 --labelfilter $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$MODELDATASET"
count++
done