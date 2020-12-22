import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List
import random
import datetime


import time
import re
from torch.utils.tensorboard import SummaryWriter
logging.basicConfig(level=logging.INFO)

FILE = "./data/en-fra.txt"

#writer = SummaryWriter("runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


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
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]


def collate(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len

class GRU_ENCODER(nn.Module):
    def __init__(self, embedding_dim, vocab_dim, hidden_dim):
        super(GRU_ENCODER, self).__init__()

        self.embedder = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim)

        #self.decoder = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, h):

        y, h = self.gru(self.embedder(x), h)

        return y, h

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

class GRU_DECODER(nn.Module):
    def __init__(self, embedding_dim, vocab_dim, hidden_dim, output_dim):
        super(GRU_DECODER, self).__init__()

        self.hidden_dim = hidden_dim

        self.embedder = nn.Embedding(vocab_dim, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.decoder = nn.Linear(hidden_dim, output_dim)

        self.vocab_dim = vocab_dim

    def forward(self, x, h):

        y, h = self.gru(self.relu(self.embedder(x)), h)

        return self.decoder(y), h

    def zero_hidden(self):
        return torch.zeros(1,1, self.hidden_dim)

    def generate(self, h, out=None, batchSize=1, constrained=True, max_len=20):

        if out != None:
            final_seq = torch.tensor([]).to(device)
            y2 = start_word*torch.ones(batchSize).long().unsqueeze(0).to(device)
            '''
            if constrained == True:
                print("constrained")
                y2, h = self.forward(y2, h)
                final_seq = torch.cat((final_seq, y2))
            else:
                print("not constrained")
                for q in range(len(out)):
                    y2, h = self.forward(y2,h)
                    final_seq = torch.cat((final_seq, y2))
                    y2 = torch.argmax(y2.squeeze(), 1).unsqueeze(0)
                '''
            #print("y2 shape: {}".format(y2.shape))
            for q in range(len(out)):

                y2, h = self.forward(y2, h)

                final_seq = torch.cat((final_seq, y2))

                if constrained==False:
                    y2 = torch.argmax(y2.squeeze(), 1).unsqueeze(0)
                else:
                    y2 = out[q].unsqueeze(0)

            return final_seq.squeeze()

        if out == None:
            final_seq = torch.tensor([]).to(device)
            y2 = start_word*torch.ones(batchSize).long().unsqueeze(0).to(device)
            for q in range(max_len):

                y2, h = self.forward(y2, h)

                final_seq = torch.cat((final_seq, y2))

                y2 = torch.argmax(y2.squeeze(), 1).unsqueeze(0)

            return final_seq

    def translate(self, h, k=10, max_len=20):

        start = start_word * torch.ones(k).long().unsqueeze(0).to(device)
        final_seq = torch.tensor([]).long().to(device)
        final_seq = torch.cat((final_seq, start))
        i = 0

        mask = torch.ones(k).long().to(device)
        probas = torch.ones(k).to(device)

        return self.beam_search(h, final_seq, i, k, max_len, mask, probas)


    def beam_search(self, h, final_sequence, i, k, max_len, mask, probas):

        if i<max_len:
            i += 1

            y, h = self.forward(final_sequence[-1].unsqueeze(0), h)
            y = y.squeeze()
            h = h.squeeze()
            y = torch.nn.functional.softmax(y, 1)
            y_flat = (probas*y.permute(1,0)).permute(1,0).flatten(0)
            values, indexes = torch.topk(y_flat, k)
            probas = probas*values/sum(probas*values)
            probas[torch.nonzero((mask == 0))] = 1000000

            words = torch.fmod(indexes, self.vocab_dim)*mask

            mask = mask*(words != 1)#.long() #1 is eos

            h_new = h[torch.floor_divide(indexes, self.vocab_dim)]
            final_sequence = final_sequence.permute(1,0)[torch.floor_divide(indexes, self.vocab_dim)].permute(1,0)

            final_sequence = torch.cat((final_sequence, words.unsqueeze(0)))

            return self.beam_search(h_new.unsqueeze(0), final_sequence, i, k, max_len, mask, probas)

        else :

            return final_sequence


class State(nn.Module):
    def __init__(self, encoder, decoder, optimizer):
        super(State, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.epoch = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)

batch_size = 150

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=100)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=100)

train_loader = DataLoader(datatrain, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(datatest, collate_fn=collate, batch_size=batch_size, shuffle=True, drop_last=True)


latent_size = 700
embedding_size = 164
encoder = GRU_ENCODER(embedding_size, len(vocEng), latent_size)
decoder = GRU_DECODER(embedding_size, len(vocFra), latent_size, vocFra.__len__())

loss = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=1e-3)

start_word = 2
end_word = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)
loss.to(device)

savepath = Path("seq_gen2.pch")
if savepath.is_file() :
    with savepath.open("rb") as fp:
        state = State(encoder, decoder, optimizer)
        params = torch.load(fp)
        state.load_state_dict(params)
else:
    state = State(encoder, decoder, optimizer)

writer = SummaryWriter("runs/traduction/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#======Parameters=====
epochs = 10
max_len = 20
prop_constrained = 0.8
def train():
    for i in range(epochs):
        avg_l = 0
        j = 0
        for eng_sent, _, french_sent, _ in train_loader:
            j += 1
            eng_sent = eng_sent.to(device).long()
            french_sent = french_sent.to(device).long()
            #print("padded shape: {}".format(padded.shape))
            h = state.encoder.initHidden(batch_size).to(device)
            y, h = state.encoder(eng_sent, h)
            if random.uniform(0,1) <= prop_constrained:
                constrained = True
            else:
                constrained = False

            final_seq = state.decoder.generate(h, french_sent, batch_size, constrained)
            if j%100 == 0:
                eng_s = vocEng.getwords(eng_sent[:,0])
                french_s = vocFra.getwords(torch.argmax(final_seq, dim=2)[:,0])
                writer.add_text("english_sentence",' '.join(map(str,eng_s)),i)
                writer.add_text("traduced_sentence", ' '.join(map(str,french_s)),i)
                print("phrase en anglais: {}".format(eng_s))
                print("phrase traduite: {}".format(french_s))

            l = loss(final_seq.view(-1, final_seq.shape[2]), french_sent.view(-1))
            avg_l += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        avg_l /= j
        print('Epoch: ', i, '\t train loss : ', avg_l)

        avg_l = 0
        j = 0
        with torch.no_grad():
            for eng_sent, _, french_sent, _ in test_loader:
                j += 1

                eng_sent = eng_sent.to(device).long()
                french_sent = french_sent.to(device).long()
                h = state.encoder.initHidden(batch_size).to(device)
                y, h = state.encoder(eng_sent, h)

                if random.uniform(0, 1) <= prop_constrained:
                    constrained = True
                else:
                    constrained = False

                final_seq = state.decoder.generate(h, french_sent, batch_size, constrained)
                if j%100 == 0:
                    eng_s = vocEng.getwords(eng_sent[:,0])
                    french_s = vocFra.getwords(torch.argmax(final_seq, dim=2)[:,0])
                    writer.add_text("english_sentence test",' '.join(map(str,eng_s)),i)
                    writer.add_text("traduced_sentence test", ' '.join(map(str,french_s)),i)
                    print("phrase en anglais test : {}".format(eng_s))
                    print("phrase traduite test: {}".format(french_s))
                l = loss(final_seq.view(-1, final_seq.shape[2]), french_sent.view(-1))
                avg_l += l.item()
            avg_l /= j
            print('Epoch: ', i, '\tTrain loss: ', avg_l)
            print()
            writer.add_scalar('Loss/Test', avg_l, i)
        # writer.add_scalar('Loss/Train', avg_l, i)

def Train():

    for i in range(epochs):

        with torch.no_grad():
            avg_l = 0
            j = 0
            for x in test_loader:
                j += 1

                x1 = x[0].to(device)
                h = state.encoder.initHidden(batch_size).to(device)
                for q in range(len(x1)):

                    y, h = state.encoder(x1[q].unsqueeze(0), h)

                x2 = x[2].to(device)

                final_seq = state.decoder.generate(h, x2, batch_size)

                l = 0
                for q in range(len(final_seq)):
                    l += loss(final_seq[q], x2[q])

                avg_l += l.item()

            avg_l /= j
            print('Epoch: ', i, '\tTest loss: ', avg_l)
            #writer.add_scalar('Loss/Test', avg_l, i)


        avg_l = 0
        j = 0
        for x in train_loader:
            j += 1

            x1 = x[0].to(device)
            h = state.encoder.initHidden(batch_size).to(device)
            for q in range(len(x1)):
                y, h = state.encoder(x1[q].unsqueeze(0), h)

            x2 = x[2].to(device)

            if random.uniform(0, 1) <= prop_constrained:
                constrained = True
            else:
                constrained = False

            final_seq = state.decoder.generate(h, x2, batch_size, constrained)

            l = 0
            for q in range(len(final_seq)):

                l += loss(final_seq[q], x2[q])

            avg_l += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        avg_l /= j
        print('Epoch: ', i, '\tTrain loss: ', avg_l)
        print()
        #writer.add_scalar('Loss/Train', avg_l, i)

        with savepath.open("wb") as fp:
            state.epoch = i + 1
            torch.save(state.state_dict(), fp)



def Generate(loader):
    k = 10
    x = next(iter(loader))[0].permute(1,0)[0].unsqueeze(1).repeat(1, k)

    print(vocEng.getwords(x.permute(1,0)[0].tolist()))
    x = x.to(device)

    h = state.encoder.initHidden(k).to(device)
    for i in range(len(x)):

        y, h = state.encoder(x[i].unsqueeze(0), h)

    translations = state.decoder.translate(h,k)

    for i in translations.permute(1,0):
        for j in vocFra.getwords(i.tolist()):
            if j == 'EOS':
                break
            if j != 'SOS' and j!= 'PAD':
                print(j, end=' ')
        print()
    print()


train()
writer.close()
Generate(train_loader)
Generate(test_loader)

