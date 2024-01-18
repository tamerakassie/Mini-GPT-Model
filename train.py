#imports
import wget
import torch
import torch.nn as nn
from torch.nn import functional as F

#Training SCript Hyperparameters


#Dataloader For Text Corpus
def get_text(url):
    path = wget.download(url)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

path_url =  'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
text = get_text(path_url)


#Encode the data



