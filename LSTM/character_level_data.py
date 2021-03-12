import torch
import torch.nn as nn
from CustomLSTM import CustomLSTM
from urllib.request import urlretrieve
import re
import sys
import numpy as np
import random
import html
from torch.utils.data import TensorDataset

def get_char_data():
    # Downloading dataset
    url_book="http://mek.oszk.hu/00500/00501/00501.htm"
    urlretrieve(url_book, 'book.html')
    text = open("book.html", encoding='latin-1').read().lower()

    # Removing HTML tags
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    no_tags = tag_re.sub('', text)
    text = html.escape(no_tags) 

    print('Number of total characters:', len(text))

    chars = sorted(list(set(text)))
    print('Number of unique characters:', len(chars))

    # char - number and inverse dictionaries
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print ("Index - character pairs:", indices_char)

    # sequence size
    maxlen = 40
    # step size between two consecutive sequence in text
    step = 3 
    sentences = []
    next_chars = []

    # prepearing training data set (next_chars are desired outputs)
    for i in range(0, len(text)-maxlen, step):
        sentences.append(text[i:i+maxlen])
        next_chars.append(text[i+maxlen])

    print('Nr. of training data points:', len(sentences)) 
    rand_ind = 2837
    print('A random example of training point:', sentences[rand_ind], next_chars[rand_ind])

    # converting training data into numeric data
    X = np.zeros((len(sentences), maxlen, len(chars)))
    y = np.zeros((len(sentences), len(chars)))

    # one-hot encoding
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence): 
            X[i,t,char_indices[char]] = 1
        y[i,char_indices[next_chars[i]]] = 1

    print ("Shape of training tensor:", X.shape)
    print ("Shape of test tensor:", y.shape)
    
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    