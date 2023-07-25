import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd


def train_model(train_loader, val_loader, nb_features, nb_classes, device, epochs_till_now=0,  nb_epochs=50, batch_size=32):
    class NeuralNet(nn.Module):

        def __init__(self, in_features=4, out_features=3):
            super().__init__()
            self.fc1 = nn.Linear(in_features=in_features,
                                 out_features=120)
            self.fc2 = nn.Linear(in_features=120,
                                 out_features=84)
            self.fc3 = nn.Linear(in_features=84,
                                 out_features=32)
            self.fc4 = nn.Linear(in_features=32,
                                 out_features=out_features)

        def forward(self, X):
            X = F.relu(self.fc1(X))
            X = F.relu(self.fc2(X))
            X = F.relu(self.fc3(X))
            return F.softmax(self.fc4(X),0)

    model = NeuralNet(nb_features, nb_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epochs_till_now, epochs_till_now+nb_epochs):
        acc = 0
        for j, batch in enumerate(train_loader):
            if batch is None:
                break
            X_train, y_train = batch
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = acc + accuracy_score(torch.argmax(y_pred,1).cpu().detach(), y_train.cpu().detach())

        if i % 2 == 0:
            acc = acc/j
            print(f'epoch: {i} -> loss: {loss}')
            print("batch {} train acc {}".format(j, acc))

    for j, batch in enumerate(val_loader):
        if batch is None:
            break
        X, Y = batch

        with torch.no_grad():
            output = model(X)
            acc = accuracy_score(torch.argmax(output,1).cpu().detach(),Y.cpu().detach())
            print("batch {} acc {}".format(j,acc))


    return model, i


def load_data(data_path=None, nb_classes=10):
    TARGET_COLUMN = 'median_house_value'


    df = pd.read_csv(data_path) if data_path else pd.read_csv("data/housing.csv")
    df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    inputs = df.drop([TARGET_COLUMN, 'ocean_proximity'],axis=1).to_numpy()
    targets = df[[TARGET_COLUMN]].to_numpy()

    est = KBinsDiscretizer(n_bins=nb_classes, encode='ordinal', strategy='uniform')
    est.fit(targets)
    labels = est.transform(targets)

    scaler = StandardScaler()
    scaler.fit(inputs)
    features = scaler.transform(inputs)

    return torch.FloatTensor(features), torch.LongTensor(labels).squeeze(1)


def get_model(device="cuda", shuffle = False, data_path=None, model="housing.pth",
              batch_size=32, epochs_till_now=0, nb_classes=10):

    x,y = load_data(data_path,nb_classes)
    dataset = TensorDataset(x.to(device), y.to(device))

    split = len(x)*4//5
    train_ds, val_ds = random_split(dataset, [split, len(x)-split])
    val_loader = DataLoader(val_ds, batch_size)

    model_path = os.path.join("models", model)
    if not os.path.exists(model_path):
        print('{} does not exists. training from scratch'.format(model_path))
        train_loader = DataLoader(train_ds, batch_size, shuffle=shuffle)
        nb_features = x.shape[1]
        model, epochs_till_now  = train_model(train_loader, val_loader,nb_features, nb_classes, device, epochs_till_now, batch_size)

        torch.save({
            'epochs': epochs_till_now,
            'model': model,
        }, model_path)

    ckpt = torch.load(model_path)
    model = ckpt['model']
    model.to(device)
    model.eval()

    return val_loader, model
