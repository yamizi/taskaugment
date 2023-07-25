#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_R50_perf_ATE"
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
ATTACK="multilabel"
MODEL="resnet50"
NAME="vanilla_baseline_sgd"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
SUFFIX="_adv-multilabel_adv-sgd"

#VANILLA
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 0 --labelfilter "class_object#$ATTACK" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/class_object#${ATTACK}${suffix}.pt" --batch_size 256 --labelfilter $MODEL --attack_target $ATTACK  --dataset_dir $DATASETDIR --output_dir "/mnt/lscratch/users/sghamizi/models/baseline/$DATASET"

#JIGSAW
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 0 --labelfilter "class_object#$ATTACK" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/class_object#${ATTACK}${suffix}_jigsaw-1.0.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir $DATASETDIR --output_dir "/mnt/lscratch/users/sghamizi/models/baseline/$DATASET"

#ROTATION
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 0 --labelfilter "class_object#$ATTACK" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/class_object#${ATTACK}${suffix}_rot-1.0.pt" --loss_rot 1 --batch_size 256 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir $DATASETDIR --output_dir "/mnt/lscratch/users/sghamizi/models/baseline/$DATASET"

#AUTOENCODER
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 0 --labelfilter "class_object#$ATTACK" --model "multi_task_$MODEL" 4 --record_roc 0 --weights_file "best/${MODEL}/class_object#${ATTACK}${suffix}_ae-1.0.pt" --loss_ae 1 --batch_size 256 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir $DATASETDIR --output_dir "/mnt/lscratch/users/sghamizi/models/baseline/$DATASET"
