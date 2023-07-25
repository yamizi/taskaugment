CUDA=2
DATASET="NIH"
DATA_FOLDER="/raid/data/datasets/NIH/images-224"
EPSILON="0.004"
NORM="Linf"
NAME="metrics"
WORKSPACE="aaai22_health"
BATCH=32
NB_BATCH=80 #80*64 inputs = 5120
MAX_ITER=25
MODEL_LABELS="densenet121-res224-all##densenet121-res224-nih##densenet121-res224-chex##densenet121-res224-pc##densenet121-res224-mimic_nb##densenet121-res224-mimic_ch##densenet121-res224-rsna"
EXPERIMENT="threat_force1"
EXPERIMENT="threat_forceall"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --experiment $EXPERIMENT --model_labels $MODEL_LABELS
