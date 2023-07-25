#!/bin/bash -l
### df-ulhpc -i to get storage quota
### squeue -u <username>
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "NIH_R50_sex-1_Consolidation"
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

cd ~/TopK/
##pip install --user virtualenv
##virtualenv venv
##source venv/bin/activate
pip install --user  -r requirements.txt
# This should report a single GPU (over 4 available per gpu node)

CUDA_VISIBLE_DEVICES=0 python experiments/train_xrayvision.py --labelfilter "Consolidation"  -name train-xray --dataset nih --loss_sex -1 --model multi_task_resnet50 --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/xray
