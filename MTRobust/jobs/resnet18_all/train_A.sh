CUDA=0
ARCH=resnet18

# D main task
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train As --class_to_test As --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train Ad --class_to_test Ad --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train Ae --class_to_test Ae --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train An --class_to_test An --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train Ar --class_to_test Ar --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train Ak --class_to_test Ak --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train AK --class_to_test AK --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train AD --class_to_test AD --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train AE --class_to_test AE --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train Ap --class_to_test Ap --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01


