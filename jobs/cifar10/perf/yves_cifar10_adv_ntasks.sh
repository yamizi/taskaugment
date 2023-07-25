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
DATASETDIR="/mnt/lscratch/users/yletraon/datasets/datasets"
WEIGHT=${1:-mgda}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}
NAME="nips_ntasks"
OUTPUTDIR="/mnt/lscratch/users/yletraon/models/baseline_${WEIGHT}/maxup/$DATASET"
BATCHSIZE=384
SEED=10


## attacking main task only


#jigsaw_rot_macro
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_jigsaw-1.0-10_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1 --loss_jigsaw 1

#jigsaw_rot
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-class_object#multilabel_jigsaw-1.0-10_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1 --loss_jigsaw 1

#rot_macro
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1

#jigsaw_macro
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_jigsaw-1.0-10_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_jigsaw 1


### attacking all tasks

#jigsaw_rot_macro
MAIN="multilabel-macro-jigsaw-rotation"
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_jigsaw-1.0-10_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1 --loss_jigsaw 1


#jigsaw_rot
MAIN="multilabel-jigsaw-rotation"
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-class_object#multilabel_jigsaw-1.0-10_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1 --loss_jigsaw 1

#rot_macro
MAIN="multilabel-macro-rotation"
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_rot-1.0-4_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_rot 1

#jigsaw_macro
MAIN="multilabel-macro-jigsaw"
WEIGHTS="/mnt/lscratch/users/yletraon/models/aux_attack/aux_cifar10_train-multi_task_resnet50-train-aux-weight-multilabel-macro_jigsaw-1.0-10_adv-multilabel_optim-sgd_cosine-0_w-mgda_a-MADRY-best.pt"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "multilabel-macro" --model "multi_task_$MODEL" --nb_secondary_labels 2 --nb_rotations 4 --permutations_jigsaw 10 --record_roc 0 --weights_file "${WEIGHTS}" --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --seed $SEED --loss_jigsaw 1

