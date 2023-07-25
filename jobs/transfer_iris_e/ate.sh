#!/bin/bash -l

while getopts m:d:e: flag
do
    case "${flag}" in
        m) MODELDATASET=${OPTARG};;
        d) DATASET=${OPTARG};;
        e) EPOCH=${OPTARG};;
    esac
done

echo "Model: $MODELDATASET";
echo "Dataset: $DATASET";
echo "Epoch: $EPOCH";

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "$DATASET_$MODELDATASET_R50_transfer_ATE"
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

ATTACK="Atelectasis"
MODELS='Atelectasis Atelectasis-Cardiomegaly Atelectasis-Pneumonia Atelectasis-Effusion Atelectasis-Consolidation Atelectasis-Edema Atelectasis-Pneumothorax'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/transfer.py" -name transfer_large --dataset $DATASET --weights_file "$MODEL/e$EPOCH.pt" --batch_size 64 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray/$MODELDATASET"
count++
done