import os
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from aptos2019.lib.preprocess import preprocess
from aptos2019.lib.dataset import Dataset
from aptos2019.lib.models.model_factory import get_model as get_mdl

def get_model(device="cuda", shuffle = False, data_path=None, model="se_resnext50_32x4d_080911",
              batch_size=32, fold=0):

    device = torch.device(device)

    args = joblib.load('models/{}/args.pkl'.format(model))


    if args.pred_type == 'classification':
        num_outputs = 5

    test_transform = transforms.Compose([
        transforms.Resize((args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # data loading code
    test_dir = preprocess(
        'test',
        args.img_size,
        scale=args.scale_radius,
        norm=args.normalize,
        pad=args.padding,
        remove=args.remove,
        img_folder=data_path)

    test_df = pd.read_csv('data/aptos2019/test.csv')
    test_img_paths = test_dir + '/' + test_df['id_code'].values + '.png'
    test_labels = np.zeros(len(test_img_paths))

    test_set = Dataset(
        test_img_paths,
        test_labels,
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)

    model_path = 'models/{}/model_{}.pth' .format(args.name, fold + 1)
    if not os.path.exists(model_path):
        print('{} does not exists.'.format(model_path))
        raise FileNotFoundError

    model = get_mdl(model_name=args.arch,
                      num_outputs=num_outputs,
                      freeze_bn=args.freeze_bn,
                      dropout_p=args.dropout_p)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    return test_loader, model
