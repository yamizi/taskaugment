#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "nih_R50_perfMadry"
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

DATASET="nih"
MAIN="Edema"
MODEL="resnet50"
NAME="adv_nih_madry200"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/nih_adv/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
AUGMENT=${3:-"_"}
TRAINATK=${8:-MADRY}
BATCHSIZE=32

PREFIX="adv-"

#ROTATION
ATTACKTASKS="Edema-rotation"
SUFFIX="_adv-Edema_optim-sgd_cosine-0_w-${WEIGHT}"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --max_eps 4 --record_all_tasks 0 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size $BATCHSIZE --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --augment ${AUGMENT}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --max_eps 4 --record_all_tasks 0 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --augment ${AUGMENT}
