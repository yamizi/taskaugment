#!/bin/bash -l
### df-ulhpc -i to get storage quota
### squeue -u <username>
### Request one GPU tasks for 4 hours - dedicate 1/4 of available cores for its management

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_Densenet_multitask_Effusion"
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
module load lang/Python/3.8.6-GCCcore-10.2.0
module load lang/Python/3.8.6-GCCcore-10.2.0

#module use /opt/apps/resif/iris/2020b/gpu/modules/all
#module load devel/PyTorch/1.9.0-fosscuda-2020b

DATASETDIR=" /work/projects/medical-generalization/datasets"
SOURCEDIR="/mnt/irisgpfs/projects/multi_task_chest_xray"
OUTPUTDIR='/mnt/lscratch/users/yletraon/models/xray'
cd ~/TopK/

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
##pip install --user virtualenv
##virtualenv venv
##source venv/bin/activate
pip install --user  -r requirements.txt
# This should report a single GPU (over 4 available per gpu node)

CUDA_VISIBLE_DEVICES=0 python experiments/train_xrayvision.py --batch_size 32 --labelfilter "Effusion"  --lr  0.001 -name train-xray-densenet --dataset chex --model multi_task_densenet --dataset_dir  $DATASETDIR --output_dir  $OUTPUTDIR
