#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_R50_perfAdv"
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
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

DATASET="aux_cifar10"
MAIN="class_object#multilabel"
MODEL="resnet50"
NAME="adv_baseline_partial"
DATASETDIR="/mnt/lscratch/users/yletraon/datasets/datasets"
SUFFIX="_adv-multilabel_optim-sgd_subset-0.1"
PREFIX="madry-partial"
WEIGHT=${1:-mgda}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}

OUTPUTDIR="/mnt/lscratch/users/yletraon/models/resnet50_mgda/aux_cifar10/best/resnet50"


STEPS=10
for SUBSET in "0.1" "0.25" "0.5" ; do
    WEIGHTCORE="class_object#multilabel"
    WEIGHTFILE="${PREFIX}-${WEIGHTCORE}_adv-multilabel_optim-sgd_subset-${SUBSET}_cosine-0_w-${WEIGHT}_a-${TRAINATK}.pt"
    CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file $WEIGHTFILE --nb_secondary_labels 2 --batch_size 256 --attack_target "multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

    WEIGHTCORE="multilabel-macro"
    WEIGHTFILE="${PREFIX}-${WEIGHTCORE}_adv-multilabel_optim-sgd_subset-${SUBSET}_cosine-0_w-${WEIGHT}_a-${TRAINATK}.pt"
    CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_resnet50" --record_roc 0 --weights_file $WEIGHTFILE --nb_secondary_labels 2 --batch_size 256 --attack_target "multilabel-macro"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

    WEIGHTCORE="class_object#multilabel_rot-1.0-4"
    WEIGHTFILE="${PREFIX}-${WEIGHTCORE}_adv-multilabel_optim-sgd_subset-${SUBSET}_cosine-0_w-${WEIGHT}_a-${TRAINATK}.pt"
    CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32 --loss_rot 1  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file $WEIGHTFILE --nb_secondary_labels 2 --batch_size 256 --attack_target "multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

    WEIGHTCORE="class_object#multilabel_jigsaw-1.0-4"
    WEIGHTFILE="${PREFIX}-${WEIGHTCORE}_adv-multilabel_optim-sgd_subset-${SUBSET}_cosine-0_w-${WEIGHT}_a-${TRAINATK}.pt"
    CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32 --loss_jigsaw 1  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file $WEIGHTFILE --nb_secondary_labels 2 --batch_size 256 --attack_target "multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
done