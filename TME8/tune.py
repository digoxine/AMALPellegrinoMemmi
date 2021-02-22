import optuna
import torch.nn as nn

import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import click
import torchvision

batch_size_train = 300
batch_size_test = 300

prop = 0.05
train_data = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

test_data = torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

class data(Dataset):


    def __init__(self, prop_kept, x):

        self.x = x
        self.prop_kept = prop_kept

    def __len__(self):
        return int(len(self.x)*self.prop_kept)

    def __getitem__(self, index):

        return self.x[index]

data_train = data(0.05, train_data)
data_test = data(0.2, test_data)

train_loader = DataLoader(data_train, batch_size=100, shuffle=True, drop_last=True)
test_loader = DataLoader(data_test, batch_size=100, shuffle=True, drop_last=True)

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.l1 = nn.Linear(28**2, 100)
        self.l2 = nn.Linear(100,100)
        self.l3 = nn.Linear(100,10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))

        return x

class NetNorm(nn.Module):
    def __init__(self):
        super(NetNorm, self).__init__()

        self.l0 = nn.BatchNorm1d(28**2)
        self.l1 = nn.Linear(28**2, 100)
        self.l2 = nn.Linear(100,100)
        self.l3 = nn.Linear(100,10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.l0(x)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))

        return x

def train(trial):
    model = Net()
    optim = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform('lr', 10**-6, 10**-2))
    loss = torch.nn.CrossEntropyLoss()

    epochs = 100

    avg_acc_test = 0

    for epoch in range(epochs):

        train_loss = 0
        train_acc = 0
        n = 0
        model.train()
        for xy in train_loader:

            x = xy[0].flatten(start_dim=1)
            y = xy[1]

            n += 1
            yhat = model(x)

            l = loss(yhat, y)
            train_loss += l.item()

            train_acc += (yhat.argmax(dim=1) == y).float().mean().item()

            optim.zero_grad()
            l.backward()
            optim.step()

        train_loss /= n
        train_acc /= n

        print('Train loss: ', train_loss, ' \tTrain acc: ', train_acc)

        test_loss = 0
        test_acc = 0
        n = 0
        entropy = []
        model.eval()
        for xy in test_loader:

            x = xy[0].flatten(start_dim=1)
            y = xy[1]

            n += 1
            yhat = model(x)

            l = loss(yhat, y)
            test_loss += l.item()

            test_acc += (yhat.argmax(dim=1) == y).float().mean().item()


        test_loss /= n
        test_acc /= n

        print('Test loss: ', test_loss, ' \tTest acc: ', test_acc)
        print()

        if epoch>=90:
            avg_acc_test += test_acc

    avg_acc_test /= 10

    return avg_acc_test


study = optuna.create_study()
study.optimize(train, n_trials=20)
print(study.best_params)
