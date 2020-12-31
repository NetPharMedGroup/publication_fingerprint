# https://github.com/vogt-m/ccbmlib/blob/88fd566be4a348dc1c7e03d045384cf80cbb3ef8/ccbmlib/preprocessing.py
# https://github.com/vogt-m/ccbmlib
# https://link.springer.com/article/10.1186/s13321-020-00456-1#ref-CR28


from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem import AllChem
import pandas as pd

def wash(sm, remove_stereo=False):
    """
    Perform a series of modifications to standardize a molecule.
    :param mol: an RDKit molecule
    :param remove_stereo: if True stereochemical information is removed
    :return: Smiles of a washed molecule object
    """
    mol = Chem.MolFromSmiles(sm)
    mol = Chem.RemoveHs(mol)
    mol = saltDisconection(mol)
    remover = SaltRemover.SaltRemover() # modified
    mol = remover.StripMol(mol)
    Chem.Cleanup(mol)
    mol, _ = NeutraliseCharges(mol)
    Chem.SanitizeMol(mol)
    if not remove_stereo:
        Chem.AssignStereochemistry(mol)
    Chem.SetAromaticity(mol)
    Chem.SetHybridization(mol)
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)
    smi = Chem.MolToSmiles(mol)
    return smi

def saltDisconection(mol):
    """
    Following instructions on MOE's (chemcomp) help webpage to create a similar
    dissociation between alkaline metals and organic atoms
    :param mol: RDKit molecule
    :return: Molecule with removed salts
    """
    mol = Chem.RWMol(mol)

    metals = [3, 11, 19, 37, 55]  # Alkaline: Li, Na, K, Rb, Cs
    organics = [6, 7, 8, 9, 15, 16, 17, 34, 35, 53]  # Organics: C, N, O, F, P, S, Cl, Se, Br, I
    bondsToDel = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.GetFormalCharge() != 0 or a2.GetFormalCharge() != 0:
            continue

        if a1.GetAtomicNum() in metals:
            if a2.GetAtomicNum() not in organics:
                continue
            if a1.GetDegree() != 1:
                continue
            bondsToDel.append(bond)

        elif a2.GetAtomicNum() in metals:
            if a1.GetAtomicNum() not in organics:
                continue
            if a2.GetDegree() != 1:
                continue
            bondsToDel.append(bond)

    for bond in bondsToDel:
        mol.RemoveBond(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())

    return Chem.Mol(mol)

def _InitialiseNeutralisationReactions():
    """ Taken from http://www.rdkit.org/docs/Cookbook.html
    """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

def NeutraliseCharges(mol, reactions=None):
    """ Taken from http://www.rdkit.org/docs/Cookbook.html
    """
    if reactions is None:
        reactions = _InitialiseNeutralisationReactions()
    replaced = False
    for i, (reactant, product) in enumerate(reactions):
        while mol.HasSubstructMatch(reactant):
            replaced = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]
    if replaced:
        return (mol, True)
    else:
        return (mol, False)


#def func(x):
#    return wash(x)


class DataPrep(object):
    def __init__(self, data, cutoff=[8,140], f=wash):
        '''
        data: pd.Series with index as cid and value as SMILES
        chembl_data: pd.Series other data if you want to combine two datasets
        cutoff: list of len 2, with cutoff[0] - lower cutoff, cutoff[1] - higher cutoff. refers to SMILES length
        f: function applied on data to standardize it        
        if your input is longer than 1e5, multiprocessing with num_cores = cpu_count()-2 with be used if available
        '''
        self.data = data.copy() 
        self.cutoff = cutoff
        self.func = f
        #self.out = ['G', 'd', 'Fe', 'Mn', 'Au', 'Be','Ba','Bi','Kr','Sr','Xe','He','Rb','Al','+4','%12','Ra','As','Ag','Mg','Ca',
        #            'Zn','Te','Li','Se','se','%11','10','Si','+3', 'te', 'p', 'b'] # just looked at the vocab.freq
        #self.out = [] # oxygen or nitrogen atoms or contained atoms besides C, N, S, O, F, Cl, Br, and H
        self.dropped_cids = dict()
        print('---------------')
        print(f'num_SMILES: {self.data.shape[0]}, size cutoffs: {self.cutoff} ')
    
    def run_wash(self):
        print('----start of wash----')
        if len(self.data) < 1e4:
            all_data = self.data.apply(self.func)
            all_data.rename("smiles", inplace=True)
            return all_data
        else:
            try:
                from multiprocessing import Pool, cpu_count
                with Pool(processes=cpu_count()-2) as pool:
                    res = pool.map(self.func, self.data)
                    all_data = pd.Series(res)
                    all_data.rename("smiles", inplace=True)
                    all_data.index = self.data.index
                    pool.close()
                    pool.join()
                return all_data
            except ImportError:
                all_data = self.data.apply(self.func)
                all_data.rename("smiles", inplace=True)
                return all_data
           
    

    def chop(self):
        
        all_data = self.run_wash()
        print('----start chop----')
        
        #short side
        mask = all_data.map(len) >= self.cutoff[0]
        self.dropped_cids[self.cutoff[0]]=sorted(list(mask[mask == False].index))
        b = sum(~mask)
        all_data = all_data[mask]

        #long side 
        mask = all_data.map(len) <= self.cutoff[1] # higher 
        self.dropped_cids[self.cutoff[1]]=sorted(list(mask[mask == False].index))
        a = sum(~mask)
        all_data = all_data[mask]
        print(f'remove {a+b} SMILES with cut-off {self.cutoff}')
        return all_data 

    # run all at once
    def fin(self):
        all_data = self.chop()
        print('---------------')
        print(f'----final num SMILES: {len(all_data)}')
        print('---------------')
        return all_data
        