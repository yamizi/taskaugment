#!/bin/bash -l
#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "R50_train3_ate"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

DATASET="chex"

while getopts m:d: flag
do
    case "${flag}" in
        m) SOURCEMODEL=${OPTARG};;
        d) DATASET=${OPTARG};;
    esac
done

echo "Model: $SOURCEMODEL";
echo "Dataset: $DATASET";

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

BASEMODEL='Cardiomegaly'

if [ $SOURCEMODEL == "Consolidation" ]
then
   MODELS='Edema Effusion Pneumonia Pneumothorax'
elif [ $SOURCEMODEL == "Edema" ]
then
  MODELS='Effusion Pneumonia Pneumothorax'
elif [ $SOURCEMODEL == "Effusion" ]
then
  MODELS='Pneumonia Pneumothorax'
else
   MODELS='Pneumothorax'
fi

DATASETDIR=" /work/projects/medical-generalization/datasets"
SOURCEDIR="/mnt/irisgpfs/projects/multi_task_chest_xray"
OUTPUTDIR='/mnt/lscratch/users/yletraon/models/xray'


for MODEL in $MODELS
do
if [ $MODEL == $SOURCEMODEL ]
then
   echo "Second and third models are equal to $MODEL"
else
   CUDA_VISIBLE_DEVICES=0 python  experiments/train_xrayvision.py -name train-xray-triplet --labelfilter "$BASEMODEL-$SOURCEMODEL-$MODEL" --dataset $DATASET --model multi_task_resnet50 --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR
fi

done

