#!/bin/bash -l
### df-ulhpc -i to get storage quota
### squeue -u <username>
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "NIH_R50_ATE_MT_adv_Atelectasis"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
#module use /opt/apps/resif/iris/2019b/gpu/modules/all
module load lang/Python/3.8.6-GCCcore-10.2.0

#module use /opt/apps/resif/iris/2020b/gpu/modules/all
#module load devel/PyTorch/1.9.0-fosscuda-2020b

THREADS=${4:-8}
FINE=${5:-0}

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
# This should report a single GPU (over 4 available per gpu node)
CUDA_VISIBLE_DEVICES=0 python experiments/train_xrayvision.py --labelfilter "Atelectasis" --loss_age 1 --attack_target "Atelectasis" --batch_size 32 -name train-nih-adv --max_eps 4  --steps 4  --weight_strategy ${1:-ew} --force_cosine ${2:-0} --algorithm ${3:-MADRY} --dataset nih --model multi_task_resnet50 --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/xray_adv --num_epochs 250 --threads $THREADS  --fine_tune $FINE