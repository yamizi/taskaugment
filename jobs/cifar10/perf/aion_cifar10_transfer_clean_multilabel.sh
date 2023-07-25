#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CIFAR10_R50_perfClean"
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
NAME="vanilla_baseline_sgd2"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/baseline/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
SUFFIX="_optim-sgd"
PREFIX="clean_v2/"
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
TRAINATK=${3:-MADRY}

(0.0153, 0.942)
--labelfilter "class_object#multilabel" --labelfilter_surrogate "class_object#multilabel" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4


(0.6205, 0.942)
--labelfilter "class_object#multilabel" --labelfilter_surrogate "multilabel-jigsaw" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4



(0.6252, 0.942)
--labelfilter "class_object#multilabel" --labelfilter_surrogate "multilabel-rotation" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4



(0.1545, 0.6812)
--labelfilter "multilabel-rotation" --labelfilter_surrogate "class_object#multilabel" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4


(0.0195, 0.6812)
--labelfilter "multilabel-rotation" --labelfilter_surrogate "multilabel-rotation" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4

(0.2065, 0.6812)
--labelfilter "multilabel-rotation" --labelfilter_surrogate "multilabel-jigsaw" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4



(0.1344, 0.9337)
--labelfilter "multilabel-jigsaw" --labelfilter_surrogate "class_object#multilabel" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4


(0.2014, 0.9337)
--labelfilter "multilabel-jigsaw" --labelfilter_surrogate "multilabel-rotation" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_rot-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4


(0.0141, 0.9337)
--labelfilter "multilabel-jigsaw" --labelfilter_surrogate "multilabel-jigsaw" --record_all_tasks 100 --model "multi_task_resnet50" --weights_file "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --weights_file_surrogate "clean_v2/class_object#multilabel_jigsaw-1.0_optim-sgd.pt" --nb_secondary_labels 4 --dataset "aux_cifar10" --attack_target "class_object#multilabel" --batch_size 128 --record_roc 0 --output_dir "D:/models/baseline/aux_cifar10/best/resnet50/" --img_size 32 --permutations_jigsaw 4
