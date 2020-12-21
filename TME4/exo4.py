import string
import unicodedata
import torch
import datetime
from utils import read_temps, RNN, device, DataCSV, nn, torch, DataCSV_All, State, DataTXT_All
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from pathlib import Path







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))
latent_size = 64
sequence_pred_length = 20
batch_size = 256
number_classes = len(id2lettre)

def Generate(state):
    initial_sequence = 'this is unbelievable'
    h = torch.zeros(1, latent_size).to(device)
    x = torch.squeeze(one_hot(string2code(normalize(initial_sequence))[None, :], number_classes).to(device))

    h = state.model(x, h).squeeze()[-1].unsqueeze(0)

    yhat = nn.functional.softmax(state.model.decoder(h), 1)

    out_sequence = inv_one_hot(yhat)
    for j in range(sequence_pred_length):
        h = state.model(yhat, h)

        h = h.squeeze().unsqueeze(0)

        yhat = nn.functional.softmax(state.model.decode(h), 1)

        out_sequence += inv_one_hot(yhat)
    print(out_sequence)

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)




def one_hot(ind, dic_length):
    t = torch.zeros(len(ind), len(ind[0]), dic_length).to(device)
    for j in range(len(ind)):

        for i in range(len(ind[0])):
            t[j][i][ind[j][i]] = 1
    return t


def inv_one_hot(pred):
    arg_max = torch.argmax(pred).to(device).item()

    return id2lettre[arg_max]


def Train():


    data_train = DataTXT_All('data/trump_full_speech.txt', sequence_length)
    data = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)


    model = RNN(len(id2lettre), latent_size, len(id2lettre))

    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=5 * 10 ** -3)

    iterations = 15

    # GPU
    model.to(device)
    loss.to(device)

    writer = SummaryWriter("runs/exo4/runs_sequence_length" + str(sequence_length) + "seq_pred_len" + str(
        sequence_pred_length) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    savepath = Path("seq_gen" + "seq_len" + str(sequence_length) + "seq_pred_len" + str(sequence_pred_length) + str(
        latent_size) + ".pch")
    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp)
    else:
        state = State(model, optim)
    state.optim.lr = 5 * 10 ** -2
    for i in range(iterations):

        train_loss = 0
        nt = 0
        for x in data:

            if nt%1000==0:
                print(nt, end=', ')

            nt += 1

            y = x.to(device)
            x= one_hot(x, len(id2lettre)).to(device)
            h = torch.zeros(batch_size, latent_size).to(device)
            x = x.permute(1, 0, 2).to(device)

            h = state.model(x, h).to(device)

            yhat = state.model.decoder(h).transpose(0,1).to(device)

            l = loss(yhat.flatten(0,1), y.flatten())

            state.optim.zero_grad()
            l.backward()
            state.optim.step()

            train_loss += l.data.to(device).item()

        train_loss = batch_size*sequence_length*train_loss/nt

        writer.add_scalar('Loss/Train', train_loss, i)
        print()
        print('Epoch: ', i+1, '\tError train: ', train_loss)
        Generate(state)
        with savepath.open("wb") as fp:
            state.epoch = i+1
            torch.save(state,fp)


for sequence_length in range(20,25, 5):
#sequence_length = 15
    Train()







        #Generate()
        #Train()
        #Generate()
        #writer.close()