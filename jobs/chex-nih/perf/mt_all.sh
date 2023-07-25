#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_R50_perf_ALL"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=3:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
module load lang/Python/3.8.6-GCCcore-10.2.0

cd ~/TopK/

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0
pip install --user  -r requirements.txt

DATASET="chex-nih"
MODELS='Atelectasis Edema Effusion Consolidation Cardiomegaly'
MODELS='Effusion Consolidation Cardiomegaly'
MODEL="Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation"
WEIGHT="/work/projects/multi_task_chest_xray/chex_nih/chex-nih-multi_task_resnet50-train-xray-Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation_optim-sgd_cosine-0-best.pt"

count=0
for ATTACK in $MODELS
do
echo $ATTACK
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name chex_nih_large --model multi_task_resnet50 --dataset $DATASET --weights_file "$WEIGHT" --batch_size 32 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/work/projects/multi_task_chest_xray/chex-nih"
count++
done