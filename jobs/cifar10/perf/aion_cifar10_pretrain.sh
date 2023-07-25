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
NAME="adv_baseline_finetune"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
SUFFIX="_adv-multilabel_optim-sgd_subset-0.1"
PREFIX="finetune-"
STEPS=4
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}

OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline_pretrain/$DATASET"

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.1.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}


CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.5.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}


OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline_pretrain2/$DATASET"

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.1.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}


CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.5.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}



OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline_pretrain3/$DATASET"

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.1.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.1.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}


CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_adv-multilabel_optim-sgd_subset-0.5.pt" --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_jigsaw-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_jigsaw 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-jigsaw"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK} --permutations_jigsaw 4

CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel-rotation"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "pretrain_adv_baseline" --dataset "aux_cifar10" --steps $STEPS --record_all_tasks 0 --img_size 32  --labelfilter "class_object#multilabel" --model "multi_task_resnet50" --record_roc 0 --weights_file "best/resnet50/finetune-class_object#multilabel_rot-1.0-4_adv-multilabel_optim-sgd_subset-0.5.pt" --loss_rot 1 --nb_secondary_labels 4 --batch_size 256 --attack_target "class_object#multilabel"  --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR  --algorithm ${EVALATK}
