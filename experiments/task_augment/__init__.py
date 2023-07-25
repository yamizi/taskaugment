import argparse


def get_argparser(name="", train_arguments=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default=name,
                        help='If the name is not empty, an experiment on www.comet.ml will be instantiated to collect the experiments metrics. You will need to set up a comet API key')
    parser.add_argument('--output_dir', type=str, default="D:/models", help="Where the checkpoints of the models will be save and where the checkpoints will be looked for to continue the experiments")
    parser.add_argument('--labelfilter', type=str,
                        default="Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation-Pneumothorax-Pneumonia",
                        help="The classification tasks, it can be one of the pathologies or the main cifar10 class (class_object#multilabel); If multiple tasks are provided, they should be separated by '-'",
    )
    parser.add_argument('--batch_max', type=int, default=0, help='')

    parser.add_argument('--skip_layers', type=int, default=1, help='')
    parser.add_argument('--skip_batch', type=int, default=1, help='')

    parser.add_argument('--weights_file', default="", type=str)
    parser.add_argument('--weights_minsteps', type=int, default=1, help='')
    parser.add_argument('--weights_maxsteps', type=int, default=100, help='')
    parser.add_argument('--weights_step', type=int, default=5, help='')

    parser.add_argument('--dataset', type=str, default="chex")
    parser.add_argument('--dataset_dir', type=str, default="D:/datasets")
    parser.add_argument('--model', type=str, default="multi_task_resnet50")
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', default=1, type=int, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--threads', type=int, default=4, help='')

    parser.add_argument('--loss_sex', type=float, default=0, help='')
    parser.add_argument('--loss_age', type=float, default=0, help='')
    parser.add_argument('--loss_hog', type=float, default=0, help='')
    parser.add_argument('--loss_ae', type=float, default=0, help='')
    parser.add_argument('--loss_jigsaw', type=float, default=0, help='')
    parser.add_argument('--loss_rot', type=float, default=0, help='')
    parser.add_argument('--loss_detect', type=float, default=0, help='')
    parser.add_argument('--sections_jigsaw', type=int, default=16, help='')
    parser.add_argument('--permutations_jigsaw', type=int, default=10, help='')
    parser.add_argument('--nb_rotations', type=int, default=4, help='')
    parser.add_argument('--nb_secondary_labels', type=int, default=10, help='')
    parser.add_argument('--permutation', type=str, help='')

    parser.add_argument('--random_start', type=int, default=0, help='')
    parser.add_argument('--data_aug', type=bool, default=True, help='')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='')
    parser.add_argument('--label_concat', type=bool, default=False, help='')
    parser.add_argument('--label_concat_reg', type=bool, default=False, help='')
    parser.add_argument('--labelunion', type=bool, default=False, help='')

    if "xray" in name:
        default_img = 512
        output_size = 1
    else:
        default_img = 32
        output_size = 2
    parser.add_argument('--img_size', type=int, default=default_img, help='')
    parser.add_argument('--output_size', type=int, default=output_size, help='')

    parser.add_argument('--weight_strategy', type=str, default="")
    parser.add_argument('--record_all_tasks', type=int, default=0, help='')
    parser.add_argument('--record_roc', type=int, default=0, help='')

    parser.add_argument('--attack_target', type=str, default='', help='')
    parser.add_argument('--max_eps', type=int, default=8, help='')
    parser.add_argument('--step_eps', type=int, default=1, help='')
    parser.add_argument('--steps', type=int, default=10, help='')

    parser.add_argument('--data_subset', type=float, default=1.0, help='')
    parser.add_argument('--augment', type=str, default="")

    parser.add_argument('--algorithm', type=str, default="MADRY")

    if train_arguments:

        parser.add_argument('-workspace', type=str, default="yamizi", help='')

        parser.add_argument('--lr', type=float, default=0.1, help='')
        parser.add_argument('--num_epochs', type=int, default=400, help='')
        parser.add_argument('--taskweights', type=bool, default=True, help='')
        parser.add_argument('--featurereg', type=bool, default=False, help='')
        parser.add_argument('--weightreg', type=bool, default=False, help='')

        parser.add_argument('--optimizer', type=str, default="sgd", help='')
        parser.add_argument('--momentum', type=float, default=0.9, help='')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
        parser.add_argument('--main_metric', type=str, default="acc", help='')
        parser.add_argument('--random_labels', type=int, default=0, help='')

        parser.add_argument('--force_cosine', type=str, default="0")
        parser.add_argument('--cutmix', type=int, default=0)
        parser.add_argument('--pretrained', type=str, default="")
        parser.add_argument('--lr_schedule', type=str, default="cosine")
        parser.add_argument('--lr-min', default=1e-6, type=float)
        parser.add_argument('--restart', type=int, default=0)
        parser.add_argument('--swa_start', type=int, default=200)

        parser.add_argument('--save_all', type=int, default=100)
        parser.add_argument('--save_skip', type=int, default=5)
        parser.add_argument('--fine_tune', type=int, default=0)

    else:

        parser.add_argument('-workspace', type=str, default="task-augment", help='')
        parser.add_argument('--batch_limit', type=int, default=0)

    return parser
