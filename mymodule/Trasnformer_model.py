import argparse
import pickle
from collections import Counter
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import shutil

PAD = 0
MAX_LEN = 220

import math
import os

import numpy as np
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

def evaluate(model, test_loader, vocab, PAD):
    model.eval()
    total_loss = 0
    for b, sm in enumerate(test_loader):
        sm = torch.t(sm.cuda()) # (T,B)
        with torch.no_grad():
            output = model(sm) # (T,B,V)
        loss = F.nll_loss(output.view(-1, len(vocab)),
                               sm.contiguous().view(-1),
                               ignore_index=PAD)
        total_loss += loss.item()
    return total_loss / len(test_loader)


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(d_model=hidden_size, nhead=4, 
        num_encoder_layers=n_layers, num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        hidden = self.trfm(embedded, embedded) # (T,B,H)
        out = self.out(hidden) # (T,B,V)
        out = F.log_softmax(out, dim=2) # (T,B,V)
        return out # (T,B,V)

    def _encode(self, src):
        # src: (T,B)
        embedded = self.embed(src)  # (T,B,H)
        embedded = self.pe(embedded) # (T,B,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (T,B,H)
        penul = output.detach().cpu().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (T,B,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output) # (T,B,H)
        output = output.detach().cpu().numpy()
        # mean, max, first*2
        return np.hstack([np.mean(output, axis=0), np.max(output, axis=0), output[0,:,:], penul[0,:,:] ]) # (B,4H)
    
    def encode(self, src):
        # src: (T,B)
        return self._encode(src)
        
def save_checkpoint(state, is_best, size=16, filename='/tf/notebooks/code_for_pub/_logs_as_python_files/transformer_training_logs/'):
    name = filename + str(size) + '_checkpoint.pth.tar'
    torch.save(state, name)
    if is_best:
        bestname = filename + str(size) + '_model_best.pth.tar'
        #os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
        shutil.copyfile(name, bestname)

class TorchVocab(object):
    """
    :property freqs: collections.Counter
    :property stoi: collections.defaultdict
    :property itos: collections.defaultdict
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param counter: collections.Counter,
        :param max_size: int, vocabulary
        :param min_freq: int, vocabulary
        :param specials: list of str, vocabulary
        :param vectors: list of vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # special tokens
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)

    # override用
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class WordVocab(Vocab):
    def __init__(self, smiles, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for s in smiles:
            words = [x for x in s]

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)



class Seq2seqDataset(Dataset):

    def __init__(self, smiles, vocab, seq_len=297):#,transform=Randomizer()):
        self.smiles = smiles
        self.vocab = vocab
        self.seq_len = seq_len
        #self.transform = transform

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        sm = self.smiles[item]
        #sm = self.transform(sm) # List
        content = [self.vocab.stoi.get(token, self.vocab.unk_index) for token in sm]
        X = [self.vocab.sos_index] + content + [self.vocab.eos_index]
        padding = [self.vocab.pad_index]*(self.seq_len - len(X))
        X.extend(padding)
        return torch.tensor(X)


