import logging
logging.basicConfig(level=logging.INFO)

import heapq
from pathlib import Path
import gzip

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import torch.nn.functional as F
from tp7_preprocess import TextDataset
import math

# Utiliser tp7_preprocess pour générer le vocabulaire BPE et
# le jeu de donnée dans un format compact

# --- Configuration

# Taille du vocabulaire
vocab_size = 1000
MAINDIR = Path(__file__).parent

# Chargement du tokenizer

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)

def loaddata(mode):
    with gzip.open(f"{mode}-{vocab_size}.pth", "rb") as fp:
        return torch.load(fp)


test = loaddata("test")
train = loaddata("train")
TRAIN_BATCHSIZE=512
TEST_BATCHSIZE=512

writer = SummaryWriter()

# --- Chargements des jeux de données train, validation et test

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=TRAIN_BATCHSIZE, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=TEST_BATCHSIZE, collate_fn=TextDataset.collate)


class CNN(nn.Module):
    def __init__(self, layers, embedding_dim, vocab_dim, out_dim):
        super(CNN, self).__init__()

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim)

        for i in range(0, len(layers)):
            self.convs.append(nn.Conv1d(in_channels=1, kernel_size=(layers[i][0], embedding_dim), stride=layers[i][1],
                                        out_channels=layers[i][2]))
            self.pools.append(nn.MaxPool1d(kernel_size=layers[i][2], stride=layers[i][2]))

        self.linear = nn.Linear(sum([i[2] for i in layers]), out_dim)
        self.layers = layers

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.embedding(x).unsqueeze(1)

        conv = [self.activation(conv(x)).squeeze() for conv in self.convs]

        pool = [F.max_pool1d(conv[i], conv[i].size()[2]).squeeze() for i in range(len(self.pools))]

        concatened = torch.cat(pool, 1)

        return self.linear(concatened)

class CNN2(nn.Module):
    def __init__(self, layers, embedding_dim, vocab_dim, out_dim):
        super(CNN2, self).__init__()

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.embedding = nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim)

        self.activation = nn.ReLU()

        chan = embedding_dim
        for i in range(0, len(layers)):
            self.convs.append(nn.Conv1d(in_channels=chan, kernel_size=layers[0][0], stride=layers[i][1],
                                        out_channels=layers[i][2]))
            self.pools.append(nn.MaxPool1d(kernel_size=layers[i][0], stride=layers[i][1]))
            chan = layers[i][2]

        self.linear = nn.Linear(chan, out_dim)
        self.layers = layers

        self.displacementLength()

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        for i in range(len(self.convs)):
            x = self.activation(self.convs[i](x))
            x = self.pools[i](x)

        x = torch.max(x, -1)[0]
        x = self.linear(x)

        return x

    def lenOut(self, seq_len):

        for i in range(len(self.layers)):

            seq_len = math.floor((seq_len-self.layers[i][0])/self.layers[i][1]+1)
            seq_len = math.floor((seq_len-self.layers[i][0])/self.layers[i][1]+1)

        return seq_len

    def displacementLength(self):

        s = 1
        w = 0

        for i in self.layers:
            w += s*(i[0]-1)
            s *= i[1]

        self.s = s
        self.w = w

    def activations(self, x): #returns how much each element of sequence activates output

        initial_len = x.size()[1]

        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        for i in range(len(self.convs)):
            x = self.activation(self.convs[i](x))
            x = self.pools[i](x)

        x = x.permute(2,0,1)

        act = [torch.zeros(x.size()[1:]).to(device) for i in range(initial_len)]

        for i in range(len(x)):
            for j in range(self.w):

                act[i*self.s+j] += x[i]

        act = torch.stack(act)

        return act

#kernel_size = size of n-gram, out_channels
#model = CNN([(3,1,10), (7,1,10), (5,1,10)], 256, vocab_size, 2) #(kernel_size, stride, out_channels)
model = CNN2([(3,1,50), (3,2,10)], 256, vocab_size, 2) #(kernel_size, stride, out_channels)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=model.parameters(), lr=10**-4)

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss.to(device)

def Train():
    epochs = 10

    for epoch in range(epochs):

        train_loss = 0
        n = 0
        train_acc = 0
        for x in train_iter:
            n +=1

            y = x[1].to(device)

            yhat = model(x[0].to(device))

            l = loss(yhat, y)

            train_loss+=l.item()

            train_acc += (torch.argmax(yhat, 1) == y).float().mean().item()

            optim.zero_grad()
            l.backward()
            optim.step()

        train_loss /= n
        train_acc /= n

        print('Train loss: ', train_loss, '\tTrain acc: ', train_acc)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)


        val_loss = 0
        n = 0
        val_acc = 0
        for x in val_iter:
            n += 1

            y = x[1].to(device)

            yhat = model(x[0].to(device))

            l = loss(yhat, y)

            val_acc += (torch.argmax(yhat, 1) == y).float().mean().item()

            val_loss += l.item()

        val_loss /= n
        val_acc /= n

        print('Valid loss: ', val_loss, '\tValid acc: ', val_acc)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)
        writer.flush()

def activationsIdentification(length):
    iter = torch.utils.data.DataLoader(test, batch_size=64, collate_fn=TextDataset.collate)

    index_max_word = [0 for i in range(length)]
    max_seq = [0 for i in range(length)]
    max = [0 for i in range(length)]

    for x in iter:

        yhat = model.activations(x[0].to(device))

        yhat = yhat.transpose(0,1)
        for j in yhat:
            for i in range(length):

                if torch.max(j[:,i]).item()>max[i]:

                    max_seq[i] = x[0][0]

                    index_max_word[i] = torch.argmax(j[:,i])

                    max[i] = torch.max(j[:,i])

    for i in range(length):

        print('Feature: ', i, '\tMax Word: ', tokenizer.IdToPiece(max_seq[i][index_max_word[i]].item()),'\tValue: ', max[i].item())

Train()
activationsIdentification(10)


