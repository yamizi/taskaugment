import os
import torch
from torchvision import transforms, datasets
from torch.utils import data
import torch.optim as optim
import torch.nn as nn


class AlexnetTS(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 1000),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(in_features=1000, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h


def calculate_accuracy(output, labels):
    return torch.sum(output.argmax(1) == labels)

def train(model, loader, opt, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model
    model.train()

    for (images, labels) in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Training pass
        opt.zero_grad()

        output, _ = model(images)
        loss = criterion(output, labels)

        # Backpropagation
        loss.backward()

        # Calculate accuracy
        acc = calculate_accuracy(output, labels)

        # Optimizing weights
        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


# Function to perform evaluation on the trained model
def evaluate(model, loader, opt, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    # Evaluate the model
    model.eval()

    with torch.no_grad():
        for (images, labels) in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Run predictions
            output, _ = model(images)
            loss = criterion(output, labels)

            # Calculate accuracy
            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def get_model(device="cuda", shuffle = False, data_path="data/GTSRB", model_path="models/gtsrb.pth",
              batch_size=32):



    device = torch.device(device)

    data_transforms = transforms.Compose([
        transforms.Resize([112, 112]),
        transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
    ])


    if os.path.exists(model_path):
        ckpt = torch.load(model_path)
        model = ckpt['model']
        model.to(device)

        model.eval()

    else:
        train_data_path = data_path + "/Train"
        train_data = datasets.ImageFolder(root=train_data_path, transform=data_transforms)

        # Divide data into training and validation (0.8 and 0.2)
        epochs = 30

        ratio = 0.8
        n_train_examples = int(len(train_data) * ratio)
        n_val_examples = len(train_data) - n_train_examples

        train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])

        # Create data loader for training and validation

        train_loader = data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
        val_loader = data.DataLoader(val_data, shuffle=shuffle, batch_size=batch_size)
        model = AlexnetTS(43)
        model.to(device)

        optimizer =  optim.Adam(filter(lambda p: p.requires_grad,
                                           model.parameters()),
                                    lr=0.0001)

        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            print("Epoch-%d: " % (epoch))


            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate(model, val_loader, optimizer, criterion, device)


            print(train_loss, train_acc)
            print(val_loss, val_acc)

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss
        }, model_path)



    test_data = datasets.ImageFolder(root=data_path+"/Test", transform=data_transforms)
    test_loader = data.DataLoader(test_data, shuffle=shuffle, batch_size=batch_size)

    return test_loader, model


if __name__ == '__main__':
    #test_loader, model = get_model()

    #  Best model from:  https: // github.com / apsdehal / traffic - signs - recognition / tree / 75016    b9383aa8e25e3d224f93b2e551cffc1f729
    test_loader, model = get_model(model_path="models/model_best_gtsrb.pth")
