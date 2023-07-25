#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR100_R50_perfAdv"
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

DATASET="aux_cifar10"
MAIN="class_object#multilabel"
MODEL="resnet50"
NAME="adv_baseline_sgd"
  OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/c100/baseline/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
SUFFIX="_adv-multilabel_optim-sgd"
PREFIX="adv_mono/"

#VANILLA
ATTACKTASKS="$MAIN"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}${SUFFIX}.pt" --batch_size 256 --attack_target $ATTACKTASKS  --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#JIGSAW
ATTACKTASKS="$MAIN-jigsaw"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${ATTACKTASKS}_jigsaw-1.0${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#ROTATION
ATTACKTASKS="$MAIN-rotation"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${ATTACKTASKS}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

PREFIX="adv_bi/"

#JIGSAW
ATTACKTASKS="$MAIN-jigsaw"
SUFFIX="_adv-multilabel-jigsaw_optim-sgd"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0${ATTACKTASKS}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4  --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_jigsaw-1.0${SUFFIX}.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR

#ROTATION
ATTACKTASKS="$MAIN-rotation"
SUFFIX="_adv-multilabel-rotation_optim-sgd"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${ATTACKTASKS}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --nb_secondary_labels 4 --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_rot-1.0${SUFFIX}.pt" --loss_rot 1 --batch_size 256 --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR


#AUTOENCODER
#ATTACKTASKS="class_object#$MAIN-ae"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name $NAME --dataset $DATASET --steps 10 --record_all_tasks 625 --img_size 32  --labelfilter "class_object#$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/class_object#${MAIN}_ae-1.0${SUFFIX}.pt" --loss_ae 1 --batch_size 256 --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR


#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "adv_baseline_sgd2" --dataset "aux_cifar10" --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/adv_mono/class_object#multilabel_jigsaw-1.0_adv-multilabel_optim-sgd.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 128 --attack_target "class_object#multilabel"  --output_dir "D:/models/c100/baseline/aux_cifar10" --permutations_jigsaw 4
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "adv_baseline_sgd2" --dataset "aux_cifar10" --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/adv_mono/class_object#multilabel_jigsaw-1.0_adv-multilabel_optim-sgd.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 128 --attack_target "class_object#multilabel-jigsaw"  --output_dir "D:/models/c100/baseline/aux_cifar10" --permutations_jigsaw 4

#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "adv_baseline_sgd2" --dataset "aux_cifar10" --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/adv_mono/class_object#multilabel_rot-1.0_adv-multilabel_optim-sgd.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 128 --attack_target "class_object#multilabel-rotation"  --output_dir "D:/models/c100/baseline/aux_cifar10"
#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "adv_baseline_sgd2" --dataset "aux_cifar10" --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/adv_mono/class_object#multilabel_rot-1.0_adv-multilabel_optim-sgd.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 128 --attack_target "class_object#multilabel"  --output_dir "D:/models/c100/baseline/aux_cifar10"

#CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "adv_baseline_sgd2" --dataset "aux_cifar10" --steps 10 --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/adv_mono/class_object#multilabel_adv-multilabel_optim-sgd.pt" --nb_secondary_labels 4 --batch_size 128 --attack_target "class_object#multilabel"  --output_dir "D:/models/c100/baseline/aux_cifar10"
