CUDA=3
#DATASET="PC"
DATASET="NIH"
DATA_FOLDER="/raid/data/datasets/NIH/images-224"
#DATA_FOLDER="/raid/data/datasets/CheXpert-v1.0-small/train.csv##/raid/data/datasets/CheXpert-v1.0-small"
EPSILON="0.004"
NORM="Linf"
NAME="model_transfer_large"
WORKSPACE="aaai22_health"
BATCH=64
NB_BATCH=80 #80*64 inputs = 5120
MAX_ITER=25
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE
