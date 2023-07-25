#!/bin/bash -l
#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "R50_cka1"
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

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt

MODELS='Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation-Pneumothorax-Pneumonia Atelectasis Atelectasis-Cardiomegaly Atelectasis-Pneumonia Atelectasis-Effusion Atelectasis-Consolidation Atelectasis-Edema Atelectasis-Pneumothorax Pneumothorax Pneumothorax-Consolidation Pneumothorax-Pneumonia Pneumothorax-Effusion Pneumothorax-Edema Pneumothorax-Cardiomegaly Effusion Cardiomegaly-Effusion Effusion-Consolidation Effusion-Edema  Pneumonia-Effusion'
MODELS='Edema Consolidation-Edema Cardiomegaly-Edema Pneumonia-Edema'
count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/cka/cka_1.py" --cuda --skip_batch 0 --skip_layers 1  --batch_size 64 --labelfilter $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray"
count++
done

