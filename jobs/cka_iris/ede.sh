#!/bin/bash -l
#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "R50_cka_EDE"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=6:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail


while getopts m:d:s: flag
do
    case "${flag}" in
        m) MODELDATASET=${OPTARG};;
        d) DATASET=${OPTARG};;
        s) STRATEGY=${OPTARG};;
    esac
done

echo "Model: $MODELDATASET";
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

MODELS='Edema Cardiomegaly-Edema Pneumonia-Edema Consolidation-Edema Atelectasis-Edema Effusion-Edema Pneumothorax-Edema'

count=0
for MODEL in $MODELS
do
echo $MODEL
CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/cka_xp.py" -name cka_large --dataset $DATASET --weights_minsteps 10  --skip_layers 5 --skip_batch 0 --batch_size 16 --labelfilter $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --output_dir "/mnt/irisgpfs/projects/multi_task_chest_xray"
count++
done