import os
import io
import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

import torch.utils.data as data


class Corpus(object):
    def __init__(self, device):
        
        train_iter, val_iter, test_iter = WikiText2()
        self.device = device
        self.tokenizer = get_tokenizer('basic_english')
        self.counter = Counter()

        self.counter.update(self.tokenizer('<sos>'))

        for line in train_iter:
            self.counter.update(self.tokenizer(line))
        
        for line in val_iter:
            self.counter.update(self.tokenizer(line))
            
        for line in test_iter:
            self.counter.update(self.tokenizer(line))

        self.vocab = Vocab(self.counter)
        
        train_iter, val_iter, test_iter = WikiText2()
        
        self.train = self.data_process(train_iter).to(self.device)
        self.val = self.data_process(val_iter).to(self.device)
        self.test = self.data_process(test_iter).to(self.device)
    
    def data_process(self, raw_text_iter):
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                       dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


class TextDataset(data.Dataset):
    def __init__(self, text_ids, in_out_overlap = False, input_size = 4, seq_len=6, stride = 1):

        self.text_ids = text_ids
        self.overlap = in_out_overlap
        self.stride = stride
        self.input_size = input_size
        self.seq_len = seq_len
        self.output_length = seq_len - input_size

        assert self.output_length > 0, f"Input size greater than sequence length"

    def __len__(self):
        
        return int((len(self.text_ids) - self.seq_len) // self.stride + 1)
    
    def __getitem__(self, idx):
        if self.overlap:
            inp = self.text_ids[idx * self.stride : idx * self.stride + self.seq_len - 1]
            out = self.text_ids[idx * self.stride + self.input_size : idx * self.stride + self.seq_len]
        else:
            inp = self.text_ids[idx * self.stride : (idx + 1) * self.stride]
            out = self.text_ids[(idx + 1) * self.stride : (idx + 1) * self.stride + self.output_length]
            
        return inp, out
