from textloader import code2string, string2code,id2lettre
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, latent_size, eos=1, start="a", maxlen=30):
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles

    h = torch.zeros(1, latent_size).to('cuda')
    for j in range(len(start)):
        h = rnn(emb(string2code(start[j])).to('cuda'), h)
    i = torch.argmax(decoder(h)).item()
    final_sequence = [i]
    n = 0

    while final_sequence[-1] != eos and n<maxlen:
        h = rnn(emb(torch.tensor(final_sequence[-1]).unsqueeze(0)).to('cuda'),h)
        final_sequence.append(torch.argmax(decoder(h)).item())

        n += 1

    return code2string(final_sequence)

def generate_beam(rnn, emb, decoder, latent_size, eos=1, k=10, start="", maxlen=30, dict_size=97):

    h = torch.zeros(1, latent_size).to('cuda')
    for j in range(len(start)):
        h = rnn(emb(string2code(start[j])).to('cuda'), h)

    n = 0

    maximum, index = torch.topk(torch.nn.functional.softmax(decoder(h), 1), k)
    probas = maximum.squeeze()
    h = torch.stack([h.squeeze() for j in range(k)]).to('cuda')

    final_seq = index.transpose(0,1)
    final_seq_eos = []

    mask = torch.ones(k).long().to('cuda')

    count = 0
    while n<maxlen and count<k:
        n+=1

        h = rnn(emb(index.squeeze().to('cpu')).to('cuda'),h)

        maximum, index = torch.topk((probas*(torch.nn.functional.softmax(decoder(h), 1).transpose(0,1))).transpose(0,1).flatten(), k)

        probas = (probas[index//dict_size]*maximum)/torch.sum(probas)
        h = h[index//dict_size]

        final_seq = final_seq[index//dict_size]

        rnn.memory = rnn.memory[index//dict_size]

        index = index%dict_size

        mask = mask * (index != eos)  # .long() #1 is eos
        count = k-torch.count_nonzero(mask)

        probas[torch.nonzero((mask == 0))] = 1000000

        final_seq = torch.cat((final_seq, index.unsqueeze(1)), 1)

    final_sentences = []
    for i in final_seq:
        final_sentences.append(code2string(i.to('cpu').numpy()))

    return final_sentences


# p_nucleus
def p_nucleus(rnn, emb, decoder, latent_size, eos=1, k_max=10, start="", maxlen=30, dict_size=96, threshold=0.8):

    h = torch.zeros(1, latent_size).to('cuda')
    for j in range(len(start)):
        h = rnn(emb(string2code(start[j])).to('cuda'), h)

    n = 0

    maximum, index = torch.topk(torch.nn.functional.softmax(decoder(h), 1), k_max)
    probas = maximum.squeeze()
    h = torch.stack([h.squeeze() for j in range(k_max)]).to('cuda')

    final_seq = index.transpose(0,1)
    final_seq_eos = []


    count = 0
    while n<maxlen:
        n+=1

        h = rnn(emb(index.squeeze().to('cpu')).to('cuda'),h)

        k = 5
        p = 0
        while k<=k_max and p<threshold:

            s = torch.nn.functional.softmax(decoder(h), 1).flatten()/k
            maximum, index = torch.topk(s, k)
            sum_max = torch.sum(maximum)

            s = (probas * (torch.nn.functional.softmax(decoder(h), 1).transpose(0, 1))).transpose(0, 1).flatten()/sum_max

            maximum, index = torch.topk(s, k)
            k+=1
            p = torch.sum(maximum)
        #print(k)
        probas = (probas[index//dict_size]*maximum)
        h = h[index//dict_size]

        final_seq = final_seq[index//dict_size]

        rnn.memory = rnn.memory[index//dict_size]

        index = index%dict_size

        final_seq = torch.cat((final_seq, index.unsqueeze(1)), 1)

    final_sentences = []
    for i in final_seq:
        final_sentences.append(code2string(i.to('cpu').numpy()))

    return final_sentences

