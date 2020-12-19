from utils import read_temps, RNN, device, DataCSV, nn, torch, Decoder, DataCSV_All
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

sequence_length = 20
number_classes = 10
batch_size = 20 #excluding multi-city
temp_data_train = DataCSV_All('data/tempAMAL_train.csv', number_classes, sequence_length)
temp_data_test = DataCSV_All('data/tempAMAL_test.csv', number_classes, sequence_length)
data = DataLoader(temp_data_train, batch_size=batch_size, shuffle=True,drop_last=True)
data_train = DataLoader(temp_data_train, batch_size=1, shuffle=False,drop_last=True)
data_test = DataLoader(temp_data_test, batch_size=1, shuffle=False,drop_last=True)
labels = torch.tensor(np.array([i%number_classes for i in range(batch_size*number_classes)])).to(device)
labels_test = torch.tensor(np.array([i%number_classes for i in range(1*number_classes)])).to(device)
labels_train = torch.tensor(np.array([i%number_classes for i in range(1*number_classes)])).to(device)

data_sizes = temp_data_train.data.size()
latent_size = 20

model = RNN(1, latent_size, number_classes)
#decoder = Decoder(latent_size,number_classes)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=10**-3)
#optim_decoder = torch.optim.Adam(decoder.parameters(), lr=10**-4)

iterations = 1000

#GPU
model.to(device)
loss.to(device)
#decoder.to(device)

writer = SummaryWriter()

for i in range(iterations):

    for x in data:
        optim.zero_grad()
        x = x.to(device)
        x = torch.flatten(x.permute(1,0,2), start_dim=1)

        h = torch.zeros(1,batch_size*number_classes,latent_size).to(device)
        yhat = model(x,h)

        l = loss(yhat, labels)

        l.backward()
        optim.step()
    #print(yhat[0].data.to('cpu'))

    with torch.no_grad():
        model.eval()

        correct=0
        total=0
        for x in data_test:
            x = x.to(device)
            x = torch.flatten(x.permute(1, 0, 2), start_dim=1)
            h = torch.zeros(1, 1 * number_classes, latent_size).to(device)
            outputs = model(x,h)
            _,predicted =torch.max(outputs.data, 1)
            total += predicted.size(0)
            correct += (predicted == labels_test).sum().item()
        correct_test = correct/total

        correct_train = 1
        """correct=0
        total=0
        for x in data_train:
            x = x.to(device)
            x = torch.flatten(x.permute(1, 0, 2), start_dim=1)
            h = torch.zeros(1, 1 * number_classes, latent_size).to(device)
            outputs = model(x,h)
            _,predicted =torch.max(outputs.data, 1)
            total += predicted.size(0)
            correct += (predicted == labels_train).sum().item()
        correct_train = correct/total"""

        model.train()

        writer.add_scalar('Loss/Test', correct_test, i)
        writer.add_scalar('Loss/Train', correct_train, i)
        print('Epoch: ', i+1, '\tAccuracy train: ', correct_train, '\tAccuracy test: ', correct_test)



