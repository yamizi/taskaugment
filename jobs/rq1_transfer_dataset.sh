#Open file in vi edit with vi filename.sh command;
#
#type in vi :set ff=unix command;
#
#save file with :wq
CUDA=0
DATASET="PC"
DATA_FOLDER="/raid/data/datasets/CheXpert-v1.0-small/train.csv##/raid/data/datasets/CheXpert-v1.0-small"
DATA_FOLDER="/raid/data/datasets/PC/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv##/raid/data/datasets/PC//images-224/images-224"
EPSILON="0.004"
NORM="Linf"
NAME="dataset_transfer_large"
WORKSPACE="aaai22_health"
array=( "densenet121-res224-all" "resnet50" "densenet121-res224-nih" "densenet121-res224-chex" "densenet121-res224-pc" "densenet121-res224-nb" "densenet121-res224-ch" "densenet121-res224-rsna" )
BATCH=64
NB_BATCH=80 #80*64 inputs = 5120
MAX_ITER=25
#for MODEL_LABELS in "${array[@]}"
#do
#  echo $MODEL_LABELS
#	CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
#done
MODEL_LABELS="densenet121-res224-all"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="resnet50"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-nih"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-chex"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-pc"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-nb"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-ch"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
MODEL_LABELS="densenet121-res224-rsna"
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python "./experiments/runner.py" --norm $NORM --dataset $DATASET --name $NAME --max_eps $EPSILON --batch_size $BATCH --max_iter $MAX_ITER --data_folder $DATA_FOLDER --nb_batches $NB_BATCH --workspace $WORKSPACE --model_labels $MODEL_LABELS
