#!/bin/bash -l

#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "nih_R50_perfMadry"
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

DATASET="nih"
MAIN="Atelectasis"
MODEL="resnet50"
NAME="adv_nih_madry"
OUTPUTDIR="/mnt/lscratch/users/sghamizi/models/nih_adv/$DATASET"
DATASETDIR="/mnt/lscratch/users/sghamizi/datasets/datasets"
WEIGHT=${1:-ew}
EVALATK=${2:-PGD}
AUGMENT=${3:-"_"}
TRAINATK=${8:-MADRY}
BATCHSIZE=64

PREFIX="adv-"

#sex
ATTACKTASKS="Atelectasis-sex"
SUFFIX="_adv-Atelectasis_optim-sgd_cosine-0_w-${WEIGHT}"
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --max_eps 4 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_sex-1.0${SUFFIX}.pt" --loss_sex 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size $BATCHSIZE --attack_target $ATTACKTASKS --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --augment ${AUGMENT}
CUDA_VISIBLE_DEVICES=0 python  "experiments/test_model.py" -name "${NAME}_${EVALATK}" --dataset $DATASET --steps 10 --max_eps 4 --record_all_tasks 625 --img_size 32  --labelfilter "$MAIN" --model "multi_task_$MODEL" --record_roc 0 --weights_file "best/${MODEL}/${PREFIX}${MAIN}_sex-1.0${SUFFIX}.pt" --loss_sex 1 --nb_secondary_labels 4 --permutations_jigsaw 4 --batch_size $BATCHSIZE --attack_target "$MAIN" --dataset_dir $DATASETDIR --output_dir $OUTPUTDIR  --algorithm ${EVALATK} --augment ${AUGMENT}
