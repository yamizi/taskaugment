#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR100_R50_perfClean"
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
NAME="vanilla_baseline_sgd2"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/c100/baseline/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
SUFFIX="_optim-sgd"
PREFIX="clean_v2/"

#VANILLA
ATTACKTASKS="$MAIN"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS  --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#JIGSAW
ATTACKTASKS="$MAIN-jigsaw"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#ROTATION
ATTACKTASKS="$MAIN-rotation"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#AUTOENCODER
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_ae-1.0${SUFFIX}.pt" --loss_ae 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
