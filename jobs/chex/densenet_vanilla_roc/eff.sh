#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_DENSENET_perf_EFF"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=1:00:00
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

DATASET="chex"
ATTACK="Effusion"
MODELS='Effusion Cardiomegaly-Effusion Pneumonia-Effusion Effusion-Consolidation Atelectasis-Effusion Effusion-Edema Pneumothorax-Effusion'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name vanilla_roc_densenet --model multi_task_densenet --dataset $DATASET --weights_file "best/densenet/densenet-$MODEL.pt" --batch_size 32 --labelfilter $MODEL --attack_target "" --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$DATASET"
count++
done