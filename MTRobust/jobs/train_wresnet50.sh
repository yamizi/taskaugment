#!/bin/bash

CUDA=2
ARCH="wideresnet50"

TRAIN=(ns nd nD nE n)
#TEST=(sd sD sE sn s )

# s main task DONE


for t in "${TRAIN[@]}"
do
  CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train $t --class_to_test $t --step_size_schedule "[[0, 0.001], [170, 0.0001], [200, 0.00001]]" --optim adam --mt_lambda 0.1
done
