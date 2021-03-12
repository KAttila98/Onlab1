import os
from io import open

import torch
import torch.utils.data as data

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class TextDataset(data.Dataset):
    def __init__(self, text_ids, receptive_fields, sample_size=6):

        self.text_ids = text_ids
        self.receptive_fields = receptive_fields
        self.sample_size = sample_size
        self.output_length = sample_size - receptive_fields + 1

        assert self.output_length > 0, f"Receptive filed size greater than sample_size"

    def __len__(self):
        return int((len(self.text_ids) - (self.sample_size + 1)) / self.output_length) + 1
    
    @staticmethod
    def _variable(data):
        
        if torch.cuda.is_available():
            return torch.autograd.Variable(data.cuda())
        else:
            return torch.autograd.Variable(data)

    def __getitem__(self, idx):
        inp = self.text_ids[idx * self.output_length : idx * self.output_length + self.sample_size]
        out = self.text_ids[idx * self.output_length + self.receptive_fields : idx * self.output_length + self.sample_size + 1]

        return self._variable(inp), self._variable(out)
