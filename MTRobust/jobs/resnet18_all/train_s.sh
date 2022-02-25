CUDA=1                                                                          
ARCH=resnet18                                                                   
                                                                                
# s main task                                                                   
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train sEDn --class_to_test sEDn --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1                                  

#CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train dDE --class_to_test dDE --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

#CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train dDn --class_to_test dDn --step_size_schedule "[[0, 0.001], [120, 0.0001], [140,  0.00001]]" --optim adam --mt_lambda 0.1
                                 
#CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train dDEn --class_to_test dDEn --step_size_schedule "[[0, 0.001], [120, 0.0001], [140,  0.00001]]" --optim adam --mt_lambda 0.1
