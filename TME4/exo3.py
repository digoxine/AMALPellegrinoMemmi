from utils import read_temps, RNN, device

#  TODO:  Question 3 : Prédiction de séries temporelles

from utils import read_temps, RNN, device, DataCSV, nn, torch, DataCSV_All
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import datetime
for sequence_length in range(5,20, 5):
    for forecast_length in range(1,3,1):
        #sequence_length = 20 #including forecast
        number_classes = 5
        batch_size = 20 #excluding multi-city
        #forecast_length = 2
        temp_data_train = DataCSV_All('data/tempAMAL_train.csv', number_classes, sequence_length)
        temp_data_test = DataCSV_All('data/tempAMAL_test.csv', number_classes, sequence_length)
        data = DataLoader(temp_data_train, batch_size=batch_size, shuffle=True,drop_last=True)
        data_test = DataLoader(temp_data_test, batch_size=1, shuffle=False,drop_last=True)


        data_sizes = temp_data_train.data.size()
        latent_size = 10

        model = RNN(1, latent_size, 1)
        loss = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=10**-2)

        iterations = 15

        #GPU
        model.to(device)
        loss.to(device)

        writer = SummaryWriter("runs/exo3/runs"+"sequence_length"+str(sequence_length)+"forecast_length"+str(forecast_length)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        for epoch in range(iterations):

            train_loss = 0
            nt = 0
            for x in data:
                nt += 1
                optim.zero_grad()
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

                l.backward()
                optim.step()
            train_loss = train_loss/nt

            print('Test')
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

                writer.add_scalar('Loss/Test', test_loss, epoch)
                writer.add_scalar('Loss/Train', train_loss, epoch)
                print('Epoch: ', epoch, '\tError train: ', train_loss, '\tError test: ', test_loss)

        writer.close()

