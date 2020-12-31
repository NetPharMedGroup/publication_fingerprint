import os
import sys
import numpy as np
import pandas as pd
import pickle 
import dill
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl import graph
from dgl.nn.pytorch import GraphConv
import rdkit
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.utils.data import DataLoader, Dataset
from progiter import ProgIter
from sklearn.model_selection import train_test_split

#ATOM_FDIM=54

class MolDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        print('Dataset includes {:d} graphs'.format(len(graphs)))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]



ELEM_LIST = ['Br','Sr','Zn', 'Se', 'se','Ba', 'Si', 'Au', 'Ag', 'Rb', 'Bi', 'Te', 'Pt', 'Ra', 'te', 'As', \
'Cl', 'Al','Fe','Cs','Ca', 'Mg', 'Gd', 'Mn', 'Sn','B','b','S','s','c','I','F','p','P','H','N','O','unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 23 + degree, charge, is_aromatic = 39
rdkit.RDLogger.logger().setLevel(rdkit.RDLogger.CRITICAL) # turn off RDKit logger

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    ELEM_LIST = ['Br','Sr','Zn', 'Se', 'se','Ba', 'Si', 'Au', 'Ag', 'Rb', 'Bi', 'Te', 'Pt', 'Ra', 'te', 'As', \
'Cl', 'Al','Fe','Cs','Ca', 'Mg', 'Gd', 'Mn', 'Sn','B','b','S','s','c','I','F','p','P','H','N','O','unknown']
    ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1  # 23 + degree, charge, is_aromatic = 39
    return (torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()]))

def mols2graphs(mols):
    """
    inputs
      mols: a list of molecules
    outputs
      cand_graphs: a list of dgl graphs 
    """
    graphs = []
    for mol in ProgIter(mols):
        n_atoms = mol.GetNumAtoms()
        g = DGLGraph()        
        node_feats = []
        for i, atom in enumerate(mol.GetAtoms()):
            assert i == atom.GetIdx()
            node_feats.append(atom_features(atom))
        g.add_nodes(n_atoms)
        bond_src = []
        bond_dst = []
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            begin_idx = a1.GetIdx()
            end_idx = a2.GetIdx()
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
        g.add_edges(bond_src, bond_dst)
        
        g.ndata['h'] = torch.Tensor([a.tolist() for a in node_feats])
        graphs.append(g)
    return graphs

def mols2graphs_mod(mol):
    """
    inputs
      mols: single mol
    outputs
      cand_graphs: a single dgl graphs 
    """

    n_atoms = mol.GetNumAtoms()
    g = DGLGraph()        
    node_feats = []
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        node_feats.append(atom_features(atom))
    g.add_nodes(n_atoms)
    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        begin_idx = a1.GetIdx()
        end_idx = a2.GetIdx()
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
    g.add_edges(bond_src, bond_dst)

    g.ndata['h'] = torch.Tensor([a.tolist() for a in node_feats])
        
    return g


def smiles2mols(smiles):
    mols = []
    for sm in ProgIter(smiles):
        mol = get_mol(sm)
        if mol is not None:
            mols.append(mol)
        else:
            print('Could not construct a molecule:', sm)
    return mols

# splitting list using yield
def chunk(l, chunk_size):
    '''
    l input list, chunk_size = size of single lists
    needed because our graph dataset is too huge if using all mols at once
    '''
    for i in range(0, len(l), chunk_size):
        yield l[i:i+chunk_size]
        

#gcn_msg =  fn.u_mul_e('h', 'weight', 'm')
#gcn_reduce = fn.sum(msg='m', out='h')  # sum aggregation
#fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h_neigh') / fn.copy_u('h','m')

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
    
    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(fn.copy_u('h','m'), fn.sum('m', 'h'))
        h = self.linear(g.ndata['h'])
        if self.activation is not None:
            h = self.activation(h)
        return h

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims, activation_f = F.relu):
        super(GAE, self).__init__()
        layers = [GCN(in_dim, hidden_dims[0], activation_f)]
        if len(hidden_dims)>=2:
            layers = [GCN(in_dim, hidden_dims[0], activation_f)]
            for i in range(1,len(hidden_dims)):
                if i != len(hidden_dims)-1:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], activation_f))
                else:
                    layers.append(GCN(hidden_dims[i-1], hidden_dims[i], lambda x:x))
        else:
            layers = [GCN(in_dim, hidden_dims[0], lambda x:x)]
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['h']
        for conv in self.layers: # conv = TAGConv(10, 2, k=2),  res = conv(g, feat)
            h = conv(g, h)
        return h

class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj
    
def collate(samples):
    bg = dgl.batch(samples)
    return bg

class Trainer:
    def __init__(self, model, lr, optim='adam'):
        self.model = model
        if optim == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 
                                                       factor=0.1,
                                                       patience=30,
                                                       mode='min', 
                                                       min_lr=1e-6,
                                                       verbose=True)
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, g, train=True):
        
        adj = g.adjacency_matrix(transpose=False) # use weighted sum instead https://discuss.dgl.ai/t/how-to-compute-the-gradient-of-edge-with-respect-to-some-loss/860/3
        adj = adj.to_dense() # wrt dense GPU -> CPU https://discuss.dgl.ai/t/how-can-i-get-the-adjacency-matrix-of-a-dgl-block/1225/3 
        # alleviate imbalance
        pos_weight = ((adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        pos_weight = pos_weight.to('cuda')
        adj = adj.to('cuda')
        g = g.to('cuda') # i think i , not using a GPU?
        adj_logits = self.model.forward(g)
        loss = BCELoss(adj_logits, adj, pos_weight=pos_weight)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()    
        #else:
        #    self.scheduler.step(loss)
        return loss.item()

    def save(self, epoch, d):
        output_path = os.path.join(d, f'ep{epoch}')
        torch.save(self.model.state_dict(), output_path)