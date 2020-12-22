import torch.nn as nn
import torch.optim
from textloader import *
from generate import *
import logging
logging.basicConfig(level=logging.INFO)

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def inv_one_hot(pred):

    arg_max = torch.argmax(pred).to('cpu').item()

    return id2lettre[arg_max]

#  TODO:  Implémenter maskedCrossEntropy

class RNN(nn.Module):
    #  TODO:  Recopier l'implémentation du RNN (TP 4)

    def __init__(self, dim_in, dim_latent, dim_out, sequence_out_length):
        super(RNN, self).__init__()
        self.hidden_group = nn.Linear(dim_latent, dim_latent)
        self.input_group = nn.Linear(dim_in, dim_latent, bias=False)
        self.activation_function = nn.ReLU()

    def one_step(self, x, h):

        return self.activation_function(self.input_group(x)+self.hidden_group(h))

    def forward(self, x, h):

        return self.one_step(x,h)


class LSTM(nn.Module):
    #  TODO:  Implémenter un LSTM

    def __init__(self, dim_in, dim_latent):
        super(LSTM, self).__init__()

        self.forget = nn.Linear(dim_latent+dim_in, dim_latent)
        self.input = nn.Linear(dim_latent+dim_in, dim_latent)
        self.update = nn.Linear(dim_latent+dim_in, dim_latent)
        self.output = nn.Linear(dim_latent+dim_in, dim_latent)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.dim_latent = dim_latent
        self.memory = torch.rand(dim_latent).to('cuda')


    def one_step(self, x, h):
        xh = torch.cat((x,h),1)
        ft = self.sig(self.forget(xh))
        it = self.sig(self.input(xh))
        Ct = ft*self.memory +it*self.tanh(self.update(xh))
        ot = self.sig(self.output(xh))
        self.memory = Ct
        return ot*self.tanh(Ct)

    def forward(self, x, h):

        return self.one_step(x,h)

    def reset_memory(self):
        self.memory = torch.rand(self.dim_latent).to('cuda')

class GRU(nn.Module):
    #  TODO:  Implémenter un GRU
    def __init__(self, dim_in, dim_latent):
        super(GRU, self).__init__()

        self.forget = nn.Linear(dim_latent+dim_in, dim_latent)
        self.update = nn.Linear(dim_latent+dim_in, dim_latent)
        self.candidate = nn.Linear(dim_latent+dim_in, dim_latent)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def one_step(self, x, h):
        xh = torch.cat((x,h),1)

        zt = self.sig(self.update(xh))
        rt = self.sig(self.forget(xh))

        return (1-zt)\
               *h+zt\
               *self.tanh(self.candidate(\
            torch.cat((rt*h,x),1)))

    def forward(self, x, h):

        return self.one_step(x,h)

class Decoder(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):

        return self.linear(x)

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

class State:
    def __init__(self, model, optim, decoder, embedder):
        self.embedder = embedder
        self.model = model
        self.optim = optim
        self.decoder = decoder
        self.epoch = 0

sequence_length = 20 #including forecast
sequence_pred_length = 20
batch_size = 64 #excluding multi-city
embedding_size = 97

with open('trump_full_speech.txt', 'r') as file:
    text = file.read().replace('\n', '')
data_train = TextDataset(text)
data = DataLoader(data_train, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,drop_last=True)

latent_size = 97*40
number_classes = len(id2lettre)
model = GRU(embedding_size, latent_size)
decoder = Decoder(latent_size,number_classes)
loss = nn.CrossEntropyLoss(reduction='none')
optim = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()), lr=10**-3)
embedder = nn.Embedding(num_embeddings=number_classes, embedding_dim=embedding_size, padding_idx=0)

iterations = 20

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model.to(device)
decoder.to(device)
loss.to(device)

writer = SummaryWriter()


savepath = Path("seq_gen"+str(latent_size)+".pch")
if savepath.is_file() :

<<<<<<< HEAD
    with savepath.open("rb") as fp:
        state = torch.load(fp)
else:
    state = State(model, optim, decoder, embedder)
=======
    decoder = Decoder(latent_size, number_classes)
    loss = nn.CrossEntropyLoss(ignore_index=0)
    optim = torch.optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=1e-4)
    embedder = nn.Embedding(num_embeddings=number_classes, embedding_dim=embedding_size, padding_idx=0)
    savepath = Path("seq_gen" + str(latent_size) + ".pch")
    if savepath.is_file():
>>>>>>> f0a664346c9ed3147e302749d580be75a7d94b45


def Train():
    for i in range(iterations):

        train_loss = 0
        nt = 0
        seq_len = 0
        for x in data:

            if nt%10==0:
                print(nt, end=', ')

            nt += 1
            state.model.reset_memory()
            state.optim.zero_grad()
            y = x.to(device)
            x = state.embedder(x).to(device)
            h = torch.zeros(batch_size, latent_size).to(device)
            seq_len += len(x)
            seq_loss = 0#------------
            for j in range(len(x)-1):

                h = state.model(x[j], h)
                yhat = state.decoder(h)

                l = maskedCrossEntropy(yhat, y[j+1], 0, loss)
                seq_loss +=l

            seq_loss.backward()

            train_loss += seq_loss.data.to('cpu').item()

            state.optim.step()
            state.optim.zero_grad()

        train_loss = train_loss/seq_len

        writer.add_scalar('Loss/Train', train_loss, i)
<<<<<<< HEAD
=======
        rs = generate_beam(state.model, state.embedder, state.decoder, latent_size,start='He is ')
        print("rs")
        print(rs)
        writer.add_text('Text/generated', rs[0], i)
>>>>>>> f0a664346c9ed3147e302749d580be75a7d94b45
        print()
        print('Epoch: ', i+1, '\tError train: ', train_loss)

        with savepath.open("wb") as fp:
            state.epoch = i+1
            torch.save(state,fp)

<<<<<<< HEAD

#Train()
=======
    writer.close()
    return state
#Train('GRU')
>>>>>>> f0a664346c9ed3147e302749d580be75a7d94b45

start_seq = 'The world is '
#print(start_seq+generate_beam(model, embedder, decoder, latent_size,start=start_seq))
#state.model.reset_memory()
#r = generate_beam(state.model, state.embedder, state.decoder, latent_size,start=start_seq)
state = Train('LSTM')
start_seq = 'He is '
state.model.reset_memory()
<<<<<<< HEAD
r = generate_beam(state.model, state.embedder, state.decoder, latent_size,start=start_seq, maxlen=30)

for i in r:
    print(start_seq+i)
print(r)
=======
r = generate_beam(state.model, state.embedder, state.decoder. latent_size, start=start_seq)
>>>>>>> f0a664346c9ed3147e302749d580be75a7d94b45

