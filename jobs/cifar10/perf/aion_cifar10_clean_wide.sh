#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_Wide_perfClean"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=8:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Python/3.8.6-GCCcore-10.2.0

cd ~/TopK/

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0
pip install --user  -r requirements.txt

DATASET="aux_cifar10"
MAIN="class_object#multilabel"
MODEL="wideresnet70"
NAME="clean_baseline_wideresnet70"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline_w/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}
SUFFIX="_adv-multilabel_optim-sgd_cosine-0_w-$WEIGHT"
SUFFIX="_optim-sgd_cosine-0_w-$WEIGHT"
PREFIX="weight-"
BATCHSIZE=128

#VANILLA
#ATTACKTASKS="multilabel"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}${SUFFIX}.pt" --batch_size $BATCHSIZE --attack_target $ATTACKTASKS  --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#JIGSAW
ATTACKTASKS="multilabel-jigsaw"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0-4${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size $BATCHSIZE --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0-4${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#ROTATION
ATTACKTASKS="multilabel-rotation"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size $BATCHSIZE --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0-4${SUFFIX}.pt" --loss_rot 1 --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#MACRO
ATTACKTASKS="multilabel-macro"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${ATTACKTASKS}${SUFFIX}.pt" --batch_size $BATCHSIZE --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${ATTACKTASKS}${SUFFIX}.pt" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}