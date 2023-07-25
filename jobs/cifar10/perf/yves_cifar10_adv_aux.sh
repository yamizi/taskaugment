#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_R50_perfCosine"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=0:30:00
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

DATASET="aux_cifar10"
MODEL="resnet50"

DATASETDIR="/mnt/lscratch/users/yletraon/datasets/datasets"
WEIGHT=${1:-mgda}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}
COSINE=0

PREFIX="aux_cifar10_train-multi_task_resnet50-train-aux-weight-"

NAME="adv_aux_only"
OUTPUTDIR="/mnt/lscratch/users/yletraon/models/aux_attack/"

#VANILLA
#ATTACKTASKS="multilabel"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "${PREFIX}${MAIN}${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS  --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}


# aux_cifar10_train-multi_task_resnet50-train-aux-weight-class_object#multilabel_jigsaw-1.0-10_adv-jigsaw_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt
# aux_cifar10_train-multi_task_resnet50-train-aux-weight-class_object#multilabel_jigsaw-1.0-10_adv-jigsaw_optim-sgd_cosine-0_w-mgda_a-MADRY-best
# JIGSAW

ATTACKTASKS="multilabel-jigsaw"
MAIN="class_object#multilabel"
SUFFIX="_adv-jigsaw_optim-sgd_cosine-${COSINE}_w-${WEIGHT}_a-MADRY-best"

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --record_roc 0 --weights_file "${PREFIX}${MAIN}_jigsaw-1.0-10${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 10 --permutations_jigsaw 10 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --record_roc 0 --weights_file "${PREFIX}${MAIN}_jigsaw-1.0-10${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 10 --permutations_jigsaw 10 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#ROTATION
ATTACKTASKS="multilabel-rotation"
MAIN="class_object#multilabel"
SUFFIX="_adv-rotation_optim-sgd_cosine-${COSINE}_w-${WEIGHT}_a-MADRY-best"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#MACRO
ATTACKTASKS="multilabel-macro"
MAIN="multilabel-macro"
SUFFIX="_adv-macro_optim-sgd_cosine-${COSINE}_w-${WEIGHT}_a-MADRY-best"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "${PREFIX}${ATTACKTASKS}${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}"  --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "${PREFIX}${ATTACKTASKS}${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}