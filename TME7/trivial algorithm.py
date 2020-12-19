import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
import gzip
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import sentencepiece as spm
import torch.nn.functional as F
from tp7_preprocess import TextDataset

vocab_size = 1000
MAINDIR = Path(__file__).parent

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

val_size = 1000
train_size = len(train) - val_size
train, val = torch.utils.data.random_split(train, [train_size, val_size])

logging.info("Datasets: train=%d, val=%d, test=%d", train_size, val_size, len(test))
logging.info("Vocabulary size: %d", vocab_size)
train_iter = torch.utils.data.DataLoader(train, batch_size=1, collate_fn=TextDataset.collate)
val_iter = torch.utils.data.DataLoader(val, batch_size=1, collate_fn=TextDataset.collate)
test_iter = torch.utils.data.DataLoader(test, batch_size=1, collate_fn=TextDataset.collate)

count_train = [0,0,0]
count_val = [0,0,0]
count_test = [0,0,0]
l1 = 0
for i in train_iter:
    count_train[i[1][0]] += 1
    l1 += 1

l2 = 0
for i in val_iter:
    count_val[i[1][0]] += 1
    l2 += 1

l3 = 0
for i in test_iter:
    count_test[i[1][0]] += 1
    l3 += 1


print([i/l1 for i in count_train])
print([i/l2 for i in count_val])
print([i/l3 for i in count_test])
