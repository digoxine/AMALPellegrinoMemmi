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

    _, i = torch.topk(decoder(h), k)
    probas = torch.ones(k).to('cuda')

    final_sequences = [i[0][j].unsqueeze(0) for j in range(k)]
    stop = False
    hs = [h for i in range(k)]
    while n < maxlen and stop == False:
        temp = torch.tensor([]).to('cuda')
        for j in range(len(final_sequences)):
            h_new = rnn(emb(final_sequences[j][-1].unsqueeze(0).to('cpu')).to('cuda'),hs[j])
            temp = torch.cat((temp, decoder(h_new)))
            hs[j] = h_new


        temp = (probas*temp.transpose(0,1)).transpose(0,1)
        probs, top = torch.topk(temp.flatten(0), k)

        probas = probs*probas[top//dict_size]

        seq = [0 for j in range(k)]
        new_char = [0 for j in range(k)]
        for j in range(len(top)):
            seq[j] = top[j].item()//dict_size
            new_char[j] = top[j].item()%dict_size

        hs_new = [0 for j in range(k)]
        new_final_seq = [0 for j in range(k)]
        for j in range(k):
            hs_new[j] = hs[seq[j]]
            new_final_seq[j] = torch.cat((final_sequences[seq[j]],torch.tensor(new_char[j]).unsqueeze(0).to('cuda')))
        hs = hs_new
        final_sequences = new_final_seq

        n += 1

    final_sequences_char = []
    for i in final_sequences:

        final_sequences_char.append(code2string(i.to('cpu').numpy()))

    return final_sequences_char


# p_nucleus
def p_nucleus(rnn, emb, decoder, latent_size, eos=1, k_max=10, start="", maxlen=30, dict_size=96, threshold=0.8):

    h = torch.zeros(1, latent_size).to('cuda')
    for j in range(len(start)):

        h = rnn(emb(string2code(start[j])).to('cuda'), h)
    n = 0

    temp = h[0]
    temp = temp / torch.sum(temp)

    k_temp = 1
    p_total = 0
    while p_total < threshold:
        _, top = torch.topk(temp, k_temp)
        p_total = 0
        for j in top:

            p_total += temp[j].item()
        k_temp += 1

    _, i = torch.topk(decoder(h), k_temp)
    hs = [h for i in range(k_temp)]
    final_sequences = [i[0][j].unsqueeze(0) for j in range(k_temp)]
    stop = False
    while n < maxlen and stop == False:
        temp = torch.tensor([]).to('cuda')
        for j in range(len(final_sequences)):
            h_new = rnn(emb(final_sequences[j][-1].unsqueeze(0).to('cpu')).to('cuda'),hs[j])
            temp = torch.cat((temp, decoder(h_new)))
            hs[j] = h_new

        _, top = torch.topk(temp.flatten(0), k_temp)

        temp = h[0]
        temp = temp / torch.sum(temp)
        k_temp =  1
        p_total = 0
        while p_total < threshold:
            _, top = torch.topk(temp, k_temp)
            p_total = 0
            for j in top:
                p_total += temp[j].item()
            k_temp += 1

        seq = [0 for j in range(k_temp)]
        new_char = [0 for j in range(k_temp)]
        for j in range(len(top)):
            seq[j] = top[j].item()//dict_size
            new_char[j] = top[j].item()%dict_size

        new_final_seq = [0 for j in range(k_temp)]
        hs_new = [0 for j in range(k_temp)]
        for j in range(k_temp):
            hs_new[j] = hs[seq[j]]
            new_final_seq[j] = torch.cat((final_sequences[seq[j]],torch.tensor(new_char[j]).unsqueeze(0).to('cuda')))
        hs = hs_new

        final_sequences = new_final_seq

        n += 1

    final_sequences_char = []
    for i in final_sequences:

        final_sequences_char.append(code2string(i.to('cpu').numpy()))

    return final_sequences_char

