#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_R50_perfClean_Macro"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=4:00:00
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
MODEL="resnet50"
NAME="vanilla_baseline_weight"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline_weight/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}
SUFFIX="_optim-sgd_cosine-0_w-$WEIGHT"
PREFIX="weight-"
RECORD=625
ATTACKTASKS="multilabel-macro"

#MACRO
SUFFIX="_optim-sgd_cosine-0_w-ew"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

SUFFIX="_optim-sgd_cosine-0_w-mgda"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}


SUFFIX="_optim-sgd_cosine-0_w-imtl"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

SUFFIX="_optim-sgd_cosine-0_w-gv"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

SUFFIX="_optim-sgd_cosine-0_w-pcg"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$ATTACKTASKS" --model "multi_task_$MODEL" --nb_secondary_labels 2 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}multilabel-macro${SUFFIX}.pt" --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}

#AUTOENCODER
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks $RECORD --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_ae-1.0${SUFFIX}.pt" --loss_ae 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK}
