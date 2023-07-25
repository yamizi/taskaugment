import argparse

def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_argparser(name="", train_arguments=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default=name,
                        help='If the name is not empty, an experiment on www.comet.ml will be instantiated to collect the experiments metrics. You will need to set up a comet API key')
    parser.add_argument('--output_dir', type=str, default="D:/models", help="Where the checkpoints of the models will be save and where the checkpoints will be looked for to continue the experiments")
    parser.add_argument('--labelfilter', type=str,
                        default="Atelectasis-Edema-Effusion-Cardiomegaly-Consolidation-Pneumothorax-Pneumonia",
                        help="The classification tasks, it can be one of the pathologies or the main cifar10 class (multilabel); If multiple tasks are provided, they should be separated by '-'",
    )

    parser.add_argument('--weights_file', default="", type=str, help="The absolute path to the preatrained model. If not a checkpoint in output_dir for the same architecture will be searched and loaded")

    parser.add_argument('--dataset', type=str, default="chex", choices=['aux_robin_train','aux_cifar10', 'aux_imagenetR' ,'aux_cifar10_train', 'chex', 'nih', "pc", 'chex-nih'])
    parser.add_argument('--dataset_dir', type=str, default="D:/datasets", help="The path to the root folder of datasets. ")
    parser.add_argument('--model', type=str, default="multi_task_resnet50", help="The model architecture to use",
                        choices=['multi_task_resnet50','multi_task_resnet50_ss','multi_task_wide2810', 'multi_task_wideresnet', 'multi_task_resnet18', 'multi_task_densenet',
                                 'resnet50',"densenet"])
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--cuda', default=1, type=int, help='')
    parser.add_argument('--shuffle', type=bool, default=True, help='')
    parser.add_argument('--threads', type=int, default=4, help='Number of workers for dataloaders')

    parser.add_argument('--loss_sex', type=float, default=0, help='If the model includes the sex auxiliary task, the default weighr of the task.')
    parser.add_argument('--loss_age', type=float, default=0, help='If the model includes the age auxiliary task, the default weighr of the task.')
    parser.add_argument('--loss_depth', type=float, default=0,
                        help='Compute on the fly depth of the image')

    parser.add_argument('--loss_jigsaw', type=float, default=0,  help='If the model includes the jigsaw auxiliary task, the default weighr of the task.')
    parser.add_argument('--sections_jigsaw', type=int, default=16, help='The number of pieces the image will be split into')
    parser.add_argument('--permutations_jigsaw', type=int, default=10, help='The size of permutation matrix for the jigsaw operation (=output size of the jigsaw task)')

    parser.add_argument('--loss_hog', type=float, default=0, help='Not fully implemented for now')
    parser.add_argument('--loss_ae', type=float, default=0, help='Not fully implemented for now')
    parser.add_argument('--loss_detect', type=float, default=0, help='Not fully implemented for now')

    parser.add_argument('--loss_rot', type=float, default=0,
                        help='If the model includes the rotation auxiliary task, the default weighr of the task.')
    parser.add_argument('--nb_rotations', type=int, default=4, help='The number of possible angles for the rotation operation (=output size of the rotation task)')

    parser.add_argument('--nb_secondary_labels', type=int, default=2, help='The number of classes for the secondary auxiliary task (for instance macro)')

    parser.add_argument('--labelunion', type=bool, default=False, help="Whether to use all labels of a dataset or only use the one from the 'labelfilter' parameter")

    parser.add_argument('--random_start', type=int, default=1, help='How many random start the PGD attack will use at the training')
    parser.add_argument('--data_aug', type=bool, default=True, help='If the dataloader will use basic data augmentation')
    parser.add_argument('--data_aug_rot', type=int, default=45, help='Maximum amount of rotation based data-augmentation')
    parser.add_argument('--data_aug_trans', type=float, default=0.15, help='Maximum precentage of translation based data-augmentation')
    parser.add_argument('--data_aug_scale', type=float, default=0.15, help='Maximum precentage of scaling based data-augmentation')

    if "xray" in name:
        default_img = 512
        output_size = 1
    else:
        default_img = 32
        output_size = 2
    parser.add_argument('--img_size', type=int, default=default_img, help='Input image size')
    parser.add_argument('--output_size', type=int, default=output_size, help='Main classification task output shape')

    parser.add_argument('--weight_strategy', type=str, default="", choices=['ew', 'imtl', 'mgda',"pcg","gv","cag",""])
    parser.add_argument('--record_all_tasks', type=int, default=0, help="Record the metrics of gradients of the tasks every 'record_all_tasks' batches")
    parser.add_argument('--record_gradients_assets', type=int, default=0, help='Records the full gradients to Comet.ml (can significantly low down the experiment)')

    parser.add_argument('--attack_target', type=str, default='', help='The task to be attacked in the training/evaluation. Must be one of the label tasks or auxiliary task')
    parser.add_argument('--max_eps', type=int, default=8, help='Maximum epsilon size for the perturbation using inf norm, will be divided by 255')
    parser.add_argument('--step_eps', type=int, default=1, help='Maximum epsilon for each step for the perturbation using inf norm, will be divided by 255')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps of the iterative attack')

    parser.add_argument('--data_subset', type=float, default=1.0, help='Percentage of total data to be used in before splitting into train / valiation / test')
    parser.add_argument('--augment', type=str, default="", choices=['', 'stl', "ss", "gowal"], help="Additional data augmentation: 'ss': using unlabelled data, 'stl' ")



    if train_arguments:

        parser.add_argument('--algorithm', type=str, default="MADRY", choices=['MADRY', "FAST","TRADES"])

        parser.add_argument('-workspace', type=str, default="yamizi", help='The name of the workspace in CometML')

        parser.add_argument('--lr', type=float, default=0.1, help='')
        parser.add_argument('--lr_schedule', type=str, default="cosine", choices=['cyclic', "cosine","multistep", ""])
        parser.add_argument('--lr-min', default=1e-6, type=float)

        parser.add_argument('--num_epochs', type=int, default=400, help='')
        parser.add_argument('--taskweights', type=str2bool, default=True, help="Use an initialization tasks' weights based on occurence")
        parser.add_argument('--featurereg', type=bool, default=False, help='Whether to use a regularization loss on the features part (output of encoder)')
        parser.add_argument('--weightreg', type=bool, default=False, help='Whether to use a regularization loss on the weights')
        parser.add_argument('--force_cosine', type=str, default="0",help='Whether to use a cosine regularization loss on the gradients')

        parser.add_argument('--optimizer', type=str, default="sgd", help='', choices=['sgd','adam',"swa_sgd"])
        parser.add_argument('--momentum', type=float, default=0.9, help='')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
        parser.add_argument('--main_metric', type=str, default="acc", help='Which metrics to use for evaluation', choices=['auc','acc'])
        parser.add_argument('--random_labels', type=int, default=0, help='Whether to scramble the labels (for sanity check)')

        parser.add_argument('--smoothed_std', type=float, default=0, help='Whether to use smoothed augmentation, how much std')
        parser.add_argument('--cutmix', type=int, default=0, help='Whether to cutmix data augmentation')
        parser.add_argument('--maxup', type=int, default=1, help='Number of variants per batch for MaxUp augmentation')
        parser.add_argument('--pretrained', type=str, default="", help="Whether to use Torchvision pretrained models", choices=['','imagenet'])

        parser.add_argument('--restart', type=int, default=0, help="Whether to ignore checkpoints and restart training from scratch")
        parser.add_argument('--swa_start', type=int, default=200, help="If optimizer is set to 'swa_sgd', from which epoch start the stochastic weight averaging")

        parser.add_argument('--save_all', type=int, default=1, help="Until which epoch save all the checkpoints")
        parser.add_argument('--save_skip', type=int, default=10, help="Skip #epochs to save a checkpoint")
        parser.add_argument('--fine_tune', type=int, default=0, help="If 1 load only the encoder part from weights_file")

    else:

        parser.add_argument('-workspace', type=str, default="task-augment", help='The name of the workspace in CometML')
        parser.add_argument('--algorithm', type=str, default="PGD", choices=['PGD', 'AA'])
        parser.add_argument('--record_roc', type=int, default=0, help='Records the AUC roc curves')
        parser.add_argument('--batch_limit', type=int, default=0, help="Number of batches to evaluate; 0=All test batches")

    return parser
