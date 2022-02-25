CUDA=1
ARCH=resnet18
# p main task
CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train ps --class_to_test ps --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pd --class_to_test pd --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pe --class_to_test pe --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pn --class_to_test pn --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pr --class_to_test pr --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pk --class_to_test pk --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pK --class_to_test pK --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.01

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pD --class_to_test pD --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pA --class_to_test pA --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1

CUDA_VISIBLE_DEVICES=$CUDA /home/sghamizi/miniconda3/envs/salah_pytorch/bin/python mtasks_train.py --dataset taskonomy --model $ARCH --customize_class --class_to_train pE --class_to_test pE --step_size_schedule "[[0, 0.001], [120, 0.0001], [140, 0.00001]]" --optim adam --mt_lambda 0.1


