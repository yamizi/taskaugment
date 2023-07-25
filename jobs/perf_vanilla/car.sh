#!/bin/bash -l

### --dataset_dir /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir /mnt/lscratch/users/sghamizi/models/xray

DATASET="nih"
ATTACK="Cardiomegaly"
MODELS='Atelectasis-Cardiomegaly Cardiomegaly-Pneumonia Cardiomegaly-Effusion Cardiomegaly-Consolidation Cardiomegaly-Edema Pneumothorax-Cardiomegaly'

count=0
for MODEL in $MODELS
do
echo $MODEL
python "experiments/test_model.py" -name vanilla --dataset $DATASET --weights_file "best/$DATASET/$MODEL.pt" --batch_size 8 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "D:/datasets" --output_dir "D:/models"
count++
done