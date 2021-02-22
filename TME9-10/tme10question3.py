from tp9 import *

class self_attention(nn.Module):

    def __init__(self, embedding_size, out_dim, embeddings, attention_layers=3, hidden_layers=(100,100)):
        super(self_attention, self).__init__()

        self.attention_layers = attention_layers
        self. embedding_size = embedding_size

        self.kqv = nn.ModuleList()
        self.g = nn.ModuleList()

        for i in range(attention_layers):
            self.kqv.append(nn.Linear(embedding_size, embedding_size*3))
            self.g.append(nn.Linear(embedding_size,embedding_size))

        self.layers = nn.ModuleList()

        l = embedding_size
        for i in hidden_layers:
            self.layers.append(nn.Linear(l, i))
            l = i

        self.out_layer = nn.Linear(l, out_dim)

        self.activation = nn.ReLU()

        self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0)

        self.norm = nn.BatchNorm1d(embedding_size, affine=False)

        self.cls = nn.Parameter(torch.randn(embedding_size))

    def forward(self, x):

        x = self.embeddings(x)

        x = torch.cat((torch.stack([self.cls for i in range(len(x))]).unsqueeze(1),x), dim=1)

        for i in range(self.attention_layers):
            y = self.kqv[i](self.norm(x.transpose(1,2)).transpose(1,2))
            a = F.softmax(torch.bmm(y[:,:,self.embedding_size:2*self.embedding_size],y[:,:,:self.embedding_size].transpose(1,2)), dim=-1)

            f = torch.bmm(a.transpose(1,2),y[:,:,2*self.embedding_size: 3*self.embedding_size])

            x = self.activation(self.g[i](x+f))

        x = x[:,0]

        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = tensorboard.SummaryWriter()

emb_size = 50
classes = 2

word2id, embeddings, data_train, data_test = get_imdb_data(emb_size)

train_loader = DataLoader(data_train, batch_size=8, collate_fn=collate_fn, drop_last=True, shuffle=True)
test_loader = DataLoader(data_test, batch_size=8, collate_fn=collate_fn, drop_last=True, shuffle=True)

model = self_attention(emb_size, classes, torch.from_numpy(embeddings).float(), hidden_layers=(100,50), attention_layers=3)
loss = nn.CrossEntropyLoss()

optim = torch.optim.Adam(model.parameters(), lr=0.001)

model.to(device)

epochs = 100

for epoch in range(epochs):

    train_loss = 0
    train_acc = 0
    n = 0
    for d in train_loader:

        x = d[0].to(device)
        y = d[1].to(device)

        n+=1

        yhat = model(x)

        #print(yhat.size(), torch.argmax(yhat, dim=1).size(), x[1].size())

        l = loss(yhat, y)

        train_loss += l.item()

        train_acc += torch.mean((torch.argmax(yhat, dim=1) == y).float()).item()

        optim.zero_grad()
        l.backward()
        optim.step()

    train_loss /= n
    train_acc /= n

    print('Epoch: \t', epoch, 'Train loss: \t', train_loss, 'Train accuracy: \t', train_acc)

    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        n = 0
        for d in test_loader:

            x = d[0].to(device)
            y = d[1].to(device)

            n+=1

            yhat = model(x)

            #print(yhat.size(), torch.argmax(yhat, dim=1).size(), x[1].size())

            l = loss(yhat, y)

            test_loss += l.item()

            test_acc += torch.mean((torch.argmax(yhat, dim=1) == y).float()).item()


        test_loss /= n
        test_acc /= n

        print('Epoch: \t', epoch, 'Test loss: \t', test_loss, 'Test accuracy: \t', test_acc)
        print()

    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Test', test_acc, epoch)
    writer.flush()

writer.close()
