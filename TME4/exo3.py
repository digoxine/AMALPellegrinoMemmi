from utils import read_temps, RNN, device

#  TODO:  Question 3 : Prédiction de séries temporelles

from utils import read_temps, RNN, device, DataCSV, nn, torch, DataCSV_All
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

sequence_length = 20 #including forecast
number_classes = 5
batch_size = 20 #excluding multi-city
forecast_length = 2
temp_data_train = DataCSV_All('data/tempAMAL_train.csv', number_classes, sequence_length)
temp_data_test = DataCSV_All('data/tempAMAL_test.csv', number_classes, sequence_length)
data = DataLoader(temp_data_train, batch_size=batch_size, shuffle=True,drop_last=True)
data_test = DataLoader(temp_data_test, batch_size=1, shuffle=False,drop_last=True)


data_sizes = temp_data_train.data.size()
latent_size = 10

model = RNN(1, latent_size, 1)
loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=10**-3)

iterations = 1000

#GPU
model.to(device)
loss.to(device)

writer = SummaryWriter()

for i in range(iterations):

    train_loss = 0
    nt = 0
    for x in data:
        nt += 1

        x = x.to(device)
        x = torch.flatten(x.permute(1, 0, 2), start_dim=1)
        y = x[-forecast_length:].unsqueeze(2)
        x = x[:-forecast_length]
        h = torch.zeros(batch_size * number_classes, latent_size).to(device)
        h = model(x.unsqueeze(2), h)[-1]
        yhat = []
        for i in range(forecast_length):
            current_pred = model.decode(h)
            yhat.append(current_pred)
            h = model(current_pred, h)[-1]

        yhat = torch.stack(yhat)

        l = loss(yhat, y)
        train_loss += l.data.to('cpu').item()

        optim.zero_grad()
        l.backward()
        optim.step()
    train_loss = train_loss/nt


    with torch.no_grad():
        model.eval()

        test_loss=0
        n = 0
        for x in data_test:
            n  += 1
            x = x.to(device)
            x = torch.flatten(x.permute(1, 0, 2), start_dim=1)
            y = x[-forecast_length:].unsqueeze(2)
            x = x[:-forecast_length]
            h = torch.zeros(1 * number_classes, latent_size).to(device)
            h = model(x.unsqueeze(2), h)[-1]
            yhat = []
            for i in range(forecast_length):
                current_pred = model.decode(h)
                yhat.append(current_pred)
                h = model(current_pred, h)[-1]

            yhat = torch.stack(yhat)
            test_loss += loss(yhat, y).to('cpu').item()

        test_loss = test_loss/n
        model.train()

        writer.add_scalar('Loss/Test', test_loss, i)
        writer.add_scalar('Loss/Train', train_loss, i)
        print('Epoch: ', i+1, '\tError train: ', train_loss, '\tError test: ', test_loss)



