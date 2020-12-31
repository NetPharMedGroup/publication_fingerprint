import h5py
import shutil
import torch
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim.optimizer import Optimizer
import numpy as np
import pickle
import pandas as pd
from time import time as t
from collections.abc import Iterable



class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             SELU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def __init__(self, i=435, o=400, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class MolEncoder(nn.Module):

    def __init__(self, i=140, o=256, c=35):
        super(MolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
                                     SELU(inplace=True))

        self.lmbd = Lambda(435, o)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        return self.lmbd(out)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(reduction='mean')
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. -
                                    torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=256, o=140, c=35):
        super(MolDecoder, self).__init__()

        self.latent_input = nn.Sequential(nn.Linear(i, i),
                                          SELU(inplace=True))
        self.repeat_vector = Repeat(o)
        self.gru = nn.GRU(i, 501, 3, batch_first=True)
        self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
                                                          nn.Softmax(dim=-1))
                                            )

    def forward(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        return self.decoded_mean(out)



class Flatten(nn.Module):

    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)


class Repeat(nn.Module):

    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)


class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


#def reset(m):
#    if hasattr(m, 'reset_parameters'):
#        m.reset_parameters()

#def load_dataset(filename):
#    with open(filename, 'rb') as f:
#        dataset = pickle.load(f)
#    
#    return dataset


def train_model(train_loader, encoder, decoder, optimizer, dtype,
                print_every=100):
    encoder.train()
    decoder.train()

    for t, (x, y) in enumerate(train_loader):
        x_var = Variable(x.type(dtype))

        y_var = encoder(x_var)
        z_var = decoder(y_var)
        loss = encoder.vae_loss(z_var, x_var)
        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.data))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def validate_model(val_loader, encoder, decoder, dtype):
    encoder.eval()
    decoder.eval()

    avg_val_loss = 0.
    for t, (x, y) in enumerate(val_loader):
        x_var = Variable(x.type(dtype))

        y_var = encoder(x_var)
        z_var = decoder(y_var)

        avg_val_loss += encoder.vae_loss(z_var, x_var).data
    avg_val_loss /= t
    print('average validation loss: %.8f' % avg_val_loss)
    return avg_val_loss




def save_checkpoint(state, is_best, size=256, filename='/tf/notebooks/code_for_pub/_logs_as_python_files/vae_training_logs/'):
    name = filename + str(size) + '_checkpoint.pth.tar'
    torch.save(state, name)
    if is_best:
        bestname = filename + str(size) + '_model_best.pth.tar'
        shutil.copyfile(name, bestname)


def initialize_weights(m):
    if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d)):
        init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.GRU):
        for weights in m.all_weights:
            for weight in weights:
                if len(weight.size()) > 1:
                    init.xavier_uniform_(weight.data)

                    
class ProcessSMILES(object):
    '''
    can take either SMILES files on disk or loaded. Do not mix types. Must have drop_duplicates() method and type: pd.Seres()
    '''
    def __init__(self, smiles_chembl=None, smiles_dc=None):
        start = t()
        if isinstance(smiles_chembl, str):
            with open(smiles_chembl, 'rb') as f:
                self.chembl = pickle.load(f)
            if isinstance(smiles_dc, str):
                with open(smiles_dc, 'rb') as f:
                    self.dc = pickle.load(f)
            else: self.dc=smiles_dc
        elif isinstance(smiles_dc, str):
            with open(smiles_dc, 'rb') as f:
                self.dc = pickle.load(f)
            if isinstance(smiles_chembl, str):
                with open(smiles_chembl, 'rb') as f:
                    self.chembl = pickle.load(f)
            else: self.chembl=smiles_chembl
        else:
            self.chembl=smiles_chembl
            self.dc=smiles_dc
        
        # if both SMILES sets are given append them and drop druplicates()
        if all([(self.chembl is not None),(self.dc is not None)]):
            self.smiles=self.chembl.append(self.dc, ignore_index=True).drop_duplicates()
            self.max_len_smiles=self._calculate_longest()
            self.charset=self.create_charset(self.smiles) # also sets self.size_of_encoding
        
        # when only one of files is present, drop_dups()
        elif all([ (self.chembl is not None), (self.dc is None) ]):
            self.smiles=self.chembl.drop_duplicates()
            self.max_len_smiles=self._calculate_longest()
            self.charset=self.create_charset(self.smiles) # also sets self.size_of_encoding
        elif all([ (self.chembl is None), (self.dc is not None) ]):
                self.smiles=self.dc.drop_duplicates()
                self.max_len_smiles=self._calculate_longest()
                self.charset=self.create_charset(self.smiles) # also sets self.size_of_encoding
        else:
            self.smiles = None
            self.max_len_smiles = None
            self.charset = None
            self.size_of_encoding = None
        if self.smiles is not None:
            print(f'{len(self.smiles)} unique SMILES read in {round((t() - start),4)} seconds')
        
    def _calculate_longest(self):
        max_len_smiles = max(self.smiles.map(len))
        return max_len_smiles
            
        
    def one_hot_encode(self, STRING, charset=None, size_of_encoding=None, max_len_smiles=None, dtype=None):
        '''
        STRING - single SMILES string
        charset is all chars found in SMILES + ' ' to identify padded characters
        size_of_encoding should be == len(charset)
        max_len_smiles is well maxlen smiles
        DOES NOT SET CLASS VARS
        '''
        if isinstance(STRING,str):     
            if dtype is None:
                dtype=np.uint8
            if charset is None:
                if self.charset is not None:
                    charset=self.charset
                else:
                    temp=set()
                    for c in STRING:
                        temp.add(c)
                    charset=[' '] + sorted(list(temp))
                    del temp

            if size_of_encoding is None:
                if self.size_of_encoding is not None:
                    size_of_encoding=self.size_of_encoding
                else:
                    size_of_encoding=len(charset)

            if max_len_smiles is None:
                if self.max_len_smiles is not None:
                    max_len_smiles=self.max_len_smiles
                else:
                    max_len_smiles=len(STRING)
            

            return np.eye(size_of_encoding, dtype=dtype)[np.array([charset.index(x) for x in STRING.ljust(max_len_smiles)]).reshape(-1)]
        else:
            raise TypeError(f'must be a SMILES string and not {type(STRING)}')

    def one_hot_encoded(self, SMILES_series, charset=None, **kwargs):
        """
        encodes a series containing SMILES
        SETS CLASS VARS
        """
        start = t()
        if charset is None: # invoke function after initializing class Process_SMILES
            if self.charset is not None:
                charset = self.charset
                if self.size_of_encoding is None:
                    size_of_encoding = len(charset)
                else: 
                    size_of_encoding=self.size_of_encoding
                    
                if self.max_len_smiles is None:
                    max_len_smiles = max(SMILES_series.map(len))
                else:
                    max_len_smiles = self.max_len_smiles
            else:
                charset = self.create_charset(smiles=SMILES_series)
                size_of_encoding=self.size_of_encoding
                max_len_smiles = self.max_len_smiles
        
        size_of_encoding=kwargs.get('size_of_encoding') # try if provided
        if size_of_encoding is None: 
            size_of_encoding=self.size_of_encoding # try if Object has it
            if size_of_encoding is None:
                size_of_encoding=len(charset)
                self.size_of_encoding=size_of_encoding # if reached, also set self.property
        
        max_len_smiles=kwargs.get('max_len_smiles')
        if max_len_smiles is None: 
            max_len_smiles=self.max_len_smiles # try if Object has it
            if max_len_smiles is None:
                self.max_len_smiles=max(SMILES_series.map(len))
                max_len_smiles=self.max_len_smiles # if reached, also set self.property
        
        dtype=kwargs.get('dtype')
        if dtype is None: 
            dtype=np.uint8
                
        df = SMILES_series.apply(self.one_hot_encode, 
                                 charset=charset, 
                                 size_of_encoding=size_of_encoding, 
                                 max_len_smiles=max_len_smiles,
                                 dtype=dtype)
        
        num_encoded=df.shape[0]
        stacked = np.stack(df.values)
        del df
        print(f'one-hot-encoded {num_encoded} SMILES in {round((t() - start),4)} seconds')
        
        return stacked
        #return df

    def create_charset(self, smiles=None):
        """
        create the charset from a list of smiles
        also creates self.charset and self.size_of_encoding
        """
        s = set()
        if smiles is None:
            smiles = self.smiles
        for smile in smiles:
            for c in smile:
                s.add(c)
        self.charset = [' '] + sorted(list(s))
        self.size_of_encoding = len(self.charset)
        self.max_len_smiles = max(smiles.map(len))
        return self.charset

def untransform(z, charset):
    """
    convert from one-hot-encoded vector z back to SMILES
    """
    z1 = []
    if all( [(len(z.shape) == 2), isinstance(z, np.ndarray)] ): #if single string
        z_sub = np.reshape(z,(-1,(*z.shape)))
        for i in range(len(z_sub)):
            s = ""
            for j in range(len(z_sub[i])):
                oh = np.argmax(z_sub[i][j])
                s += charset[oh]
            z1.append(s.strip())
        return z1 
    elif isinstance(z, Iterable): #if iterable with strings containing strings
        for i in range(len(z)):
            s = ""
            for j in range(len(z[i])):
                oh = np.argmax(z[i][j])
                s += charset[oh]
            z1.append(s.strip())
        return z1 
