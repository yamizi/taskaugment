#!/bin/bash -l

#SBATCH --mail-user=salah.ghamizi@uni.lu
#SBATCH -J "CHEX_R50_ATE_MT_adv_AtelectasisEdema"
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

THREADS=${4:-8}
IMGSIZE=${5:-512}
FINE=${6:-0}

python3 -m venv ~/venv/salah
source ~/venv/salah/bin/activate

module load lang/Python/3.8.6-GCCcore-10.2.0
#~/venv/salah/bin/python3 -m pip install --upgrade pip

pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python experiments/train_xrayvision.py --labelfilter "Atelectasis-Edema"   --attack_target "Atelectasis" --batch_size 32 -name train-xray-adv --max_eps 4  --steps 4  --weight_strategy ${1:-ew} --force_cosine ${2:-0} --algorithm ${3:-MADRY} --dataset chex --model multi_task_resnet50 --dataset_dir  /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir  /mnt/lscratch/users/sghamizi/models/xray_adv --num_epochs 250 --threads $THREADS --img_size $IMGSIZE --fine_tune $FINE