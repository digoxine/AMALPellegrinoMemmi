from utils import read_temps, RNN, device, DataCSV, nn, torch, DataCSV_All
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

sequence_length = 15
number_classes = 10
batch_size = 30 #excluding multi-city
temp_data_train = DataCSV_All('data/tempAMAL_train.csv', number_classes, sequence_length)
temp_data_test = DataCSV_All('data/tempAMAL_test.csv', number_classes, sequence_length)
data = DataLoader(temp_data_train, batch_size=batch_size, shuffle=True,drop_last=True)
data_test = DataLoader(temp_data_test, batch_size=batch_size, shuffle=False,drop_last=True)
labels = torch.tensor(np.array([i%number_classes for i in range(batch_size*number_classes)])).to(device)
labels_test = torch.tensor(np.array([i%number_classes for i in range(batch_size*number_classes)])).to(device)

data_sizes = temp_data_train.data.size()
latent_size = 20

model = RNN(1, latent_size, number_classes)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=10**-3)

iterations = 1000

#GPU
model.to(device)
loss.to(device)
#decoder.to(device)

writer = SummaryWriter()

for i in range(iterations):

    with torch.no_grad():
        test_loss = 0
        j=0
        correct_test=0
        total=0
        for x in data_test:
            j+=1
            x = x.to(device)
            x = torch.flatten(x.permute(1, 0, 2), start_dim=1)

            h = torch.zeros(batch_size * number_classes, latent_size).to(device)

            h = model(x.unsqueeze(2), h)

            outputs = model.decode(h[-1])

            l = loss(outputs, labels)
            test_loss += l.data.to('cpu').item()

            predicted = torch.argmax(outputs.data, 1)

            total += predicted.size(0)
            correct_test += (predicted == labels_test).sum().item()
        correct_test /= total

        test_loss /= j

    correct_train = 0
    train_loss=0
    total = 0
    j=0
    for x in data:
        j+=1
        optim.zero_grad()
        x = x.to(device)

        x = torch.flatten(x.permute(1,0,2), start_dim=1)

        h = torch.zeros(batch_size*number_classes,latent_size).to(device)

        h = model(x.unsqueeze(2),h)
        yhat = model.decode(h[-1])

        l = loss(yhat, labels)

        predicted = torch.argmax(yhat.data, 1)
        total += predicted.size(0)
        correct_train += (predicted == labels).sum().item()

        l.backward()
        optim.step()
        train_loss += l.data.to('cpu').item()
    correct_train /= total

    #print(yhat[0].data.to('cpu'))
    train_loss /= j

    writer.add_scalar('Acc/Test', correct_test, i)
    writer.add_scalar('Acc/Train', correct_train, i)
    writer.add_scalar('Loss/Test', test_loss, i)
    writer.add_scalar('Loss/Train', train_loss, i)
    print('Epoch: ', i+1,' \tLoss train: ', train_loss, '\tAccuracy train: ', correct_train,' \tLoss test: ', test_loss, '\tAccuracy test: ', correct_test)




