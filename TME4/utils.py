import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)


def fill_na(mat):
    ix,iy = np.where(np.isnan(mat))
    for i,j in zip(ix,iy):
        if np.isnan(mat[i+1,j]):
            mat[i,j]=mat[i-1,j]
        else:
            mat[i,j]=(mat[i-1,j]+mat[i+1,j])/2.
    return mat

class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0

def read_temps(path):
    """Lit le fichier de températures"""
    return torch.tensor(fill_na(np.array(pd.read_csv(path).iloc[:,1:])),dtype=torch.float)

class RNN(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out):
        super(RNN, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent).to(device)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False).to(device)
        self.activation_function = nn.ReLU().to(device)

        self.decoder = nn.Linear(dim_latent, dim_out).to(device)

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):
        h_seq = [h]

        for x_u in x:

            h_seq.append(self.one_step(x_u, h_seq[-1]))

        return torch.stack(h_seq[1:]).to(device)

    def decode(self,h):
        return self.decoder(h)

class DataCSV(Dataset):

    def __init__(self, file, classes, sequence_length):
        self.data = read_temps(file)[:,:classes]
        averages = (torch.sum(self.data, dim=0)/len(self.data)).expand(len(self.data), classes)

        stds = torch.sqrt(torch.sum((self.data-averages)**2, dim=0)/len(self.data)).expand(len(self.data), classes)

        self.data = (self.data-averages)/stds
        self.sequence_length=sequence_length


    def __len__(self):
        return len(self.data)//self.sequence_length-1

    def __getitem__(self, index):
        random_start = random.randint(0,self.sequence_length)
        return self.data[index*self.sequence_length+random_start:index*self.sequence_length+random_start+self.sequence_length]

# Version all trainers
class DataCSV_All(Dataset):

    def __init__(self, file, classes, sequence_length):
        self.data = read_temps(file)[:,:classes]
        averages = (torch.sum(self.data, dim=0)/len(self.data)).expand(len(self.data), classes)

        stds = torch.sqrt(torch.sum((self.data-averages)**2, dim=0)/len(self.data)).expand(len(self.data), classes)

        self.data = (self.data-averages)/stds
        self.sequence_length=sequence_length


    def __len__(self):
        return len(self.data)-self.sequence_length

    def __getitem__(self, index):

        return self.data[index:index+self.sequence_length]

#-------------Q4-------------------
import string
import unicodedata

LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class DataTXT_All(Dataset):
    def __init__(self, file, sequence_length):
        self.sequence_length = sequence_length

        f = open(file, 'r')
        self.data = string2code(normalize(f.read()))
        f.close()

    def __len__(self):
        return len(self.data)-self.sequence_length

    def __getitem__(self, index):

        ind = self.data[index:index+self.sequence_length]

        return ind