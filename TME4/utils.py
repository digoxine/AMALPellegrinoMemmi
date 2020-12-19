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
    def __init__(self, model, optim, decoder):
        self.model = model
        self.optim = optim
        self.decoder = decoder
        self.epoch = 0

def read_temps(path):
    """Lit le fichier de températures"""
    return torch.tensor(fill_na(np.array(pd.read_csv(path).iloc[:,1:])),dtype=torch.float)

class Decoder(nn.Module):
    def __init__(self, dim_latent, dim_out):
        super(Decoder, self).__init__()
        self.decoder = torch.nn.Linear(dim_latent, dim_out)

    def forward(self, x):
        return self.decoder(x)

class RNN(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out):
        super(RNN, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()

        #decoder
        self.decoder = nn.Linear(dim_latent, dim_out)
        #self.activation_function_decoder = nn.Softmax()

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):
        h_seq = [h]

        for x_u in x:

            h_seq.append(self.one_step(x_u, h_seq[-1]))

        return torch.stack(h_seq[1:])

    def decode(self,h):
        return self.decoder(h)

"""class RNN(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out):
        super(RNN, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()

        #decoder
        self.decoder = nn.Linear(dim_latent, dim_out)
        #self.activation_function_decoder = nn.Softmax()

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):
        for i in x:
            print(x.size())

            h = torch.cat((h,self.one_step(i[:,None], h[-1])[None,:,:]),0)

        return h

    def decode(self, h):
        return self.decoder(h)"""

class RNN_forecasting(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out, forecast_length):
        super(RNN_forecasting, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()
        self.forecast_length = forecast_length

        #decoder
        self.decoder = nn.Linear(dim_latent, dim_out)
        #self.activation_function_decoder = nn.Softmax()

    def one_step(self, x, h):


        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):

        for i in x:

            h = torch.cat((h,self.one_step(i[:,None], h[-1])[None,:,:]),0)
        forecasted = self.decoder(h[-1])[None,:]
        for i in range(self.forecast_length-1):
            h = torch.cat((h,self.one_step(forecasted[-1], h[-1])[None,:,:]), 0)
            forecasted = torch.cat((forecasted, self.decoder(h[-1])[None, :]), 0)

        return forecasted[-self.forecast_length:].squeeze()



class RNN_seq_gen(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out, sequence_out_length):
        super(RNN_seq_gen, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()
        self.sequence_out_length = sequence_out_length

        #decoder
        self.decoder = nn.Linear(dim_latent, dim_out)
        #self.activation_function_decoder = nn.Softmax()

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):

        sequence_out = self.decoder(h[-1])[None, :,:]
        #x = x.permute(1,0,2)

        for i in x:

            h = torch.cat((h,self.one_step(i, h[-1])[None,:,:]),0) #?????
            sequence_out = torch.cat((sequence_out, self.decoder(h[-1])[None,:]))

        for i in range(max(0, self.sequence_out_length-1-len(x))):
            h = torch.cat((h,self.one_step(sequence_out[-1], h[-1])[None,:,:]), 0)
            sequence_out = torch.cat((sequence_out, self.decoder(h[-1])[None, :]), 0) #aaa

        return sequence_out[0:self.sequence_out_length].squeeze().permute(1,0,2)

class RNN_seq_gen2(nn.Module):

    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, dim_in, dim_latent, dim_out, sequence_out_length):
        super(RNN_seq_gen2, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)\
                                        +self.hidden_group(h))

    def forward(self, x, h):

        return self.one_step(x,h)

class Decoder(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):

        return self.linear(x)

#  TODO:  Implémenter les classes Dataset ici

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