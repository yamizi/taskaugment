#!/bin/bash -l

### --dataset_dir /mnt/lscratch/users/sghamizi/datasets/datasets --output_dir /mnt/lscratch/users/sghamizi/models/xray

DATASET="nih"
ATTACK="Consolidation"
MODELS='Cardiomegaly-Consolidation Pneumonia-Consolidation Effusion-Consolidation Atelectasis-Consolidation Consolidation-Edema Pneumothorax-Consolidation'

count=0
for MODEL in $MODELS
do
echo $MODEL
python "experiments/test_model.py" -name vanilla_large --dataset $DATASET --weights_file "best/$DATASET/$MODEL.pt" --batch_size 8 --labelfilter $MODEL --attack_target $ATTACK --dataset_dir "D:/datasets" --output_dir "D:/models"
count++
done