#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_R50_perf_pairs"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=8:00:00
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

DATASET="chex"
MODELS='Atelectasis Edema Effusion Consolidation Cardiomegaly'
MODELS_COMB="Edema-Effusion-Cardiomegaly-Consolidation Atelectasis-Edema-Effusion-Cardiomegaly Atelectasis-Edema-Effusion-Consolidation Atelectasis-Edema-Cardiomegaly-Consolidation"

count=0
for ATTACK in $MODELS
do
for MODEL in $MODELS_COMB
do
WEIGHT="/mnt/lscratch/users/sghamizi/models/xray/chex-multi_task_resnet50-train-xray-${MODEL}_optim-sgd_cosine-0-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name chex_comb_large --model multi_task_resnet50 --dataset $DATASET --weights_file "$WEIGHT" --batch_size 64 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/work/projects/multi_task_chest_xray/chex_comb"
count++
done
done