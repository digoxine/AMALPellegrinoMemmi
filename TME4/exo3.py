from utils import read_temps, RNN, device

#  TODO:  Question 3 : Prédiction de séries temporelles

from utils import read_temps, RNN, device, DataCSV, nn, torch, Decoder, RNN_forecasting, DataCSV_All
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
data_train = DataLoader(temp_data_train, batch_size=1, shuffle=False,drop_last=True)
data_test = DataLoader(temp_data_test, batch_size=1, shuffle=False,drop_last=True)


data_sizes = temp_data_train.data.size()
latent_size = 10

model = RNN_forecasting(1, latent_size, 1, forecast_length)
#decoder = Decoder(latent_size,number_classes)
loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=10**-2)
#optim_decoder = torch.optim.Adam(decoder.parameters(), lr=10**-4)

iterations = 1000

#GPU
model.to(device)
loss.to(device)
#decoder.to(device)

writer = SummaryWriter()

for i in range(iterations):

    train_loss = 0
    nt = 0
    for x in data:
        nt += 1
        optim.zero_grad()
        x = x.to(device)
        x = torch.flatten(x.permute(1, 0, 2), start_dim=1)
        y = x[-forecast_length:]
        x = x[:-forecast_length]
        h = torch.zeros(1, batch_size * number_classes, latent_size).to(device)
        yhat = model(x, h)

        l = loss(yhat, y)
        train_loss += l.data.to('cpu').item()

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
            y = x[-forecast_length:]
            x = x[:-forecast_length]
            h = torch.zeros(1, 1 * number_classes, latent_size).to(device)
            yhat = model(x,h)
            test_loss += loss(yhat, y).to('cpu').item()

        test_loss = test_loss/n
        model.train()

        writer.add_scalar('Loss/Test', test_loss, i)
        writer.add_scalar('Loss/Train', train_loss, i)
        print('Epoch: ', i+1, '\tError train: ', train_loss, '\tError test: ', test_loss)



