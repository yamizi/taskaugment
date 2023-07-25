#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "NIH_Resnet50_singletask_All"
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -C volta32
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH -p gpu
#SBATCH --mail-type=end,fail

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
#module purge || print_error_and_exit "No 'module'command"
# You probably want to use more recent gpu/CUDA-optimized software builds
module load lang/Python/3.8.6-GCCcore-10.2.0
#module load lang/Python/3.8.6-GCCcore-10.2.0

#module use /opt/apps/resif/iris/2020b/gpu/modules/all
#module load devel/PyTorch/1.8.1-fosscuda-2020b
cd ~/TopK/

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate
module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_xrayvision.py --labelfilter "Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation"  -name train-xray-densenet --dataset nih --model densenet --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/xray --batch_size 32