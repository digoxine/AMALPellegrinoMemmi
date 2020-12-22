import itertools
import logging
from tqdm import tqdm
import datetime
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
logging.basicConfig(level=logging.INFO)
from pathlib import Path

ds = prepare_dataset('org.universaldependencies.french.gsd')


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True, prop_oov_swap=0):
        self.sentences = []
        self.prop_oov_swap = prop_oov_swap

        for s in data:
            self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))

    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):

        i = torch.randperm(torch.tensor(self.sentences[ix][0]).size(0))[:int(len(self.sentences[ix][0])*self.prop_oov_swap)]
        s = torch.tensor(self.sentences[ix][0])
        s[i] = 1
        s = s.tolist()
        return s, self.sentences[ix][1]


def collate(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

prop_oov_swap = 0.8

logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)
train_data = TaggingDataset(ds.train, words, tags, True, prop_oov_swap)
dev_data = TaggingDataset(ds.validation, words, tags, True)
test_data = TaggingDataset(ds.test, words, tags, False)

logging.info("Vocabulary size: %d", len(words))

writer = SummaryWriter("runs/tagging/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate, batch_size=BATCH_SIZE)

class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.embedder = nn.Embedding(vocab_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(self.embedder(x))

        return self.decoder(h).permute(0,2,1)

class State:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.epoch = 0

model = LSTM(100, len(words), 100, len(tags))
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=10**-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss.to(device)

savepath = Path("seq_cat.pch")
if savepath.is_file():
    with savepath.open("rb") as fp:
        state = torch.load(fp)
else:
    state = State(model, optimizer)

word_test = test_data[7]
print(words.getwords(word_test[0]))
print(torch.argmax(state.model(torch.tensor(word_test[0]).unsqueeze(0).to(device)),1))

def Train():
    epochs = 20

    for i in range(epochs):
        avg_l = 0
        j = 0
        for x in train_loader:
            j += 1

            y = state.model(x[0].to(device))

            l = loss(y,x[1].to(device))
            avg_l += l.item()
            state.optimizer.zero_grad()
            l.backward()
            state.optimizer.step()
        avg_l /= j
        print('Train loss: ', avg_l)

        writer.add_scalar('Loss/Train', avg_l, i)

        with savepath.open("wb") as fp:
            state.epoch = i + 1
            torch.save(state, fp)

        with torch.no_grad():
            avg_l = 0
            avg_acc = 0
            j = 0
            for x in test_loader:
                j += 1
                y = state.model(x[0].to(device))
                l = loss(y, x[1].to(device))
                avg_l += l.item()

                label = torch.argmax(y, 1)
                avg_acc += (label == x[1].to(device)).float().mean().item()

            avg_l /= j
            avg_acc /= j
            print('Test loss: ', avg_l, '\t\t Test accuracy: ', avg_acc, i)


        writer.add_scalar('Loss/Test', avg_l, i)
        writer.add_scalar('Accuracy/Test', avg_acc, i)
        writer.flush()

Train()