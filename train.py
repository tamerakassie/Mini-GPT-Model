#imports
import wget
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TransformerDecoder
import math


#Training SCript Hyperparameters


batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 100
learning_rate = 1e-3
eval_iters = 200
dropout = 0.1
max_iters = 5000
n_layer = 6 # number of layers for the deep NN
p = 0.1
d_model = 8
d_ff = 4*d_model #From Paper
n_head = 4
head_size =8
C=8

#Dataloader For Text Corpus
def get_text(url):
    path = wget.download(url)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

path_url =  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = get_text(path_url)


#Encode the data


characters = sorted(list(set(text))) #sorts the text corpus
print(characters)
vocab_dimension = len(characters)
print(vocab_dimension ) #Dimension of 65, this comprises of all alphabets/caps/punctuation marks

#Map the characters (string) to integers (numbers)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(characters) }
itos = { i:ch for i,ch in enumerate(characters) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
print(data)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(len(data))
def get_batch(split):

    data =  train_points if split == 'train' else test_points

    ix = torch.randint(len(data)-block_size, (batch_size, ))
    x = torch.stack([data_test[i:i+block_size] for i in ix])
    y = torch.stack([data_test[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



#Training loop

model = model = TransformerDecoder(block_size, n_layer).to('cuda')
model = model.to(device)


# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


