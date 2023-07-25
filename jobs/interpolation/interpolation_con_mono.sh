#!/bin/bash -l
#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "R50_interp_con_mono"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=8:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

##  ./jobs/interpolation/interpolation_ate.sh  -d "chex"

while getopts d: flag
do
    case "${flag}" in
        d) DATASET=${OPTARG};;
    esac
done

echo "Model: $SOURCEMODEL";
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

MODEL='Consolidation'
SOURCEMODEL='Consolidation'
MODELS2='Atelectasis Cardiomegaly Effusion Consolidation Edema Pneumothorax Pneumonia'

SOURCEDIR="/mnt/irisgpfs/projects/multi_task_chest_xray"
OUTPUTDIR="/mnt/irisgpfs/projects/multi_task_chest_xray"



for TARGETMODEL in $MODELS2
do
  echo "source: $SOURCEMODEL target:$TARGETMODEL dataset:$DATASET"
  CUDA_VISIBLE_DEVICES=0 python  "experiments/task_augment/interpolation.py" -name interpolation_mono --dataset $DATASET --labelsource $SOURCEMODEL --labeltarget $TARGETMODEL --labelfilter $MODEL --dataset_dir "/mnt/lscratch/users/sghamizi/datasets/datasets" --target_dir $OUTPUTDIR  --source_dir $SOURCEDIR --batch_size 256 --cuda
done


