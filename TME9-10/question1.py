from tp9 import *

class avg_pool(nn.Module):

    def __init__(self, embedding_size, out_dim, embeddings, hidden_layers=(100,100)):
        super(avg_pool, self).__init__()

        self.layers = nn.ModuleList()

        l = embedding_size
        for i in hidden_layers:
            self.layers.append(nn.Linear(l, i))
            l = i

        self.out_layer = nn.Linear(l, out_dim)

        self.activation = nn.ReLU()

        self.embeddings = nn.Embedding.from_pretrained(embeddings, padding_idx=0)

    def forward(self, x):

        x = self.embeddings(x)

        y = x.mean(dim=1)

        for i in range(len(self.layers)):
            y = self.activation(self.layers[i](y))

        return self.out_layer(y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = tensorboard.SummaryWriter()

emb_size = 50
classes = 2

word2id, embeddings, data_train, data_test = get_imdb_data(emb_size)

train_loader = DataLoader(data_train, batch_size=128, collate_fn=collate_fn, drop_last=True, shuffle=True)
test_loader = DataLoader(data_test, batch_size=128, collate_fn=collate_fn, drop_last=True, shuffle=True)

model = avg_pool(emb_size, classes, torch.from_numpy(embeddings).float(), (200,100))
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

