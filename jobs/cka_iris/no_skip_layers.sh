#!/bin/bash -l
#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "R50_cka_ALL_NoLayerSkip"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail


DATASET="chex"
STRATEGY="EPOCHS"

echo "Dataset: $DATASET";
echo "Strategy: $STRATEGY";

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

MODELS='Atelectasis Atelectasis-Cardiomegaly Atelectasis-Pneumonia Atelectasis-Effusion Atelectasis-Consolidation Atelectasis-Edema Atelectasis-Pneumothorax Cardiomegaly Cardiomegaly-Pneumonia Cardiomegaly-Effusion Cardiomegaly-Consolidation Cardiomegaly-Edema Pneumothorax-Cardiomegaly Consolidation Consolidation-Edema Effusion Effusion-Consolidation Effusion-Edema Pneumothorax-Consolidation Pneumothorax Pneumothorax-Pneumonia Pneumothorax-Effusion Pneumothorax-Edema Pneumonia Pneumonia-Effusion Pneumonia-Consolidation Pneumonia-Edema Edema'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/cka_xp.py" -name cka_alllayers --dataset $DATASET --weights_minsteps 10  --skip_layers 1 --skip_batch 0 --batch_size 64 --labelfilter $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray" --strategy $STRATEGY
count++
done

