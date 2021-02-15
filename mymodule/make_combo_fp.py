from time import time as t
from progiter import ProgIter
import pickle
import pandas as pd
import numpy as np
import copy

class DataInput:
    '''
    fps_name should point to fps file with DrugComb ID as index. 
    low and high are cutoffs, dependent on size
    bitaverage=True only for topological fps
    '''
    def __init__(self, low=0, high=64, bitaverage=False,
                 fps_name = '/tf/notebooks/code_for_pub/fp_files/fps_transformer_64bit.pickle'):
        
        self.bitaverage = bitaverage
        self.fingerprints = dict()
        self.mapping = dict()
        self.low = low
        self.high = high
        self.fps_name = fps_name # drugID id as a column
        self.df_name = '/tf/notebooks/code_for_pub/input_files/doses_CssSyn2020_1.csv'
        self.drugs_name = '/tf/notebooks/code_for_pub/smiles_files/drugcomb_drugs_export_OCT2020.csv'
        self.smiles_filename = '/tf/notebooks/code_for_pub/smiles_files/smiles_processed_upd13March.csv' 
        
        start = t()
        with open(self.fps_name, 'rb') as f:
            self.fps = pickle.load(f)
        for i in self.fps.iterrows():
            self.fingerprints[i[0]] = i[1].values
        
        self.df = pd.read_csv(filepath_or_buffer=self.df_name,
                              sep='|',
                              engine='c',
                              lineterminator='\n',
                              quotechar='"',
                              low_memory=False,
                              usecols=['block_id','drug_row_cid','drug_col_cid','drug_col','drug_row',
                                       'cell_line_name','css_ri','synergy_zip','synergy_bliss',
                                       'synergy_loewe','synergy_hsa']
                             )
        
        self.drugs = pd.read_csv(self.drugs_name, names=['id', 'smiles', 'cid'], header=0) # oct2020 version
        self.mapping = self.drugs[['id','cid']].set_index(keys='cid', drop=True).iloc[:,0].to_dict()
        
        self.smiles = pd.read_csv(self.smiles_filename, 
                                  sep='|', 
                                  usecols=['id','smiles','cid'],
                                  index_col='id')
        self.smiles.sort_index(inplace=True)
        
        b = self.df.shape[0]
        self.df = self.df[~pd.isnull(self.df.drug_col)]
        a = self.df.shape[0]
        print(f'dropped {b-a} single drugs')
        print(f'elapsed time: {round((t() - start), 2)} seconds')
        self.mapCIDtoID()
        
    def mapCIDtoID(self):
        b = self.df.shape[0]
        self.df['drug_row_id'] = self.df['drug_row_cid']
        self.df['drug_row_id'] = self.df['drug_row_id'].map(self.mapping)
        self.df['drug_col_id'] = self.df['drug_col_cid']
        self.df['drug_col_id'] = self.df['drug_col_id'].map(self.mapping)
        
        # this is probably the most important function. It selects all the drugs we have in the fps file
        # given that drugcomb_id is used as index in it
        self.df  = self.df[ self.df.drug_row_id.isin(self.fps.index) & self.df.drug_col_id.isin(self.fps.index) ]
        
        a = self.df.shape[0]
        print(f'dropped {b-a} rows after mapping to what we have in fps')

        
    def calc_synergy(self, agg_duplicate_smiles=True):
        a = self.df.shape[0]
        t1 = t()
        if agg_duplicate_smiles==True:
            temp = self.df['drug_row'] +" "+ self.df['drug_col'] # create a new string which combines drug_row and col
            self.df['drug_row_col'] = temp.transform(str.split).transform(frozenset) # split by space, turn into set, make into a new column
            self.df2 = self.df.groupby(["drug_row_col",'cell_line_name'], as_index=False).agg( #use new column to group
                drug_row_id=   ('drug_row_id',   lambda x: min(np.unique(x))),
                drug_col_id=   ('drug_col_id',   lambda x: max(np.unique(x))),
                synergy_zip=   ('synergy_zip',   np.mean),
                synergy_bliss= ('synergy_bliss', np.mean),
                synergy_loewe= ('synergy_loewe', np.mean),
                synergy_hsa=   ('synergy_hsa',   np.mean),
                css_ri=        ('css_ri',        np.mean)
            )
            b = self.df2.shape[0]
        else:
            temp = self.df.groupby(['block_id'])[['synergy_zip','synergy_bliss','synergy_loewe','synergy_hsa']].mean()
            self.df = self.df.drop_duplicates(subset=['block_id'], 
                                                inplace=False)
            self.df = self.df.drop(columns=['drug_col',
                                              'drug_row', 
                                              'drug_row_cid',
                                              'drug_col_cid',
                                              'synergy_zip',
                                              'synergy_bliss',
                                              'synergy_loewe',
                                              'synergy_hsa'], 
                                     inplace=False)
            self.df.set_index('block_id', drop=True, inplace=True)
            self.df2 = pd.concat((self.df, temp), axis=1)
            b = self.df2.shape[0]
        
        print(f'shrank dataset from {a} to {b} rows in {round((t()-t1), 2)} seconds')
        
        if self.bitaverage:
            self.holder = np.zeros((len(self.df2), len(self.fingerprints[1][self.low:self.high])), dtype=np.float32)
            
            for e, i in enumerate(self.df2.itertuples()):
                self.holder[e] = np.mean([self.fingerprints[i.drug_row_id][self.low:self.high],
                                         self.fingerprints[i.drug_col_id][self.low:self.high]], axis=0)
        
        if not self.bitaverage:
            self.holder = np.zeros((len(self.df2), 2*len(self.fingerprints[1][self.low:self.high])), dtype=np.float32) 

            for e, i in enumerate(self.df2.itertuples()):
                self.holder[e] = np.append(self.fingerprints[i.drug_row_id][self.low:self.high],
                                         self.fingerprints[i.drug_col_id][self.low:self.high])
        
            
        ready = dict() 
        
        for x in ProgIter(['css_ri','synergy_zip','synergy_bliss','synergy_loewe','synergy_hsa'], 
                          desc='creating 5 datasets packed into dict with metrics as key names',
                          show_times=False, 
                          total=5):
            if not self.bitaverage: 
                columns = [ str(zz) + '_drugRow' if zz in range(0, int(self.holder.shape[1]/2), 1) 
                           else str(zz) + '_drugCol' for zz in range(int(self.holder.shape[1]))] # [f(x) if condition else g(x) for x in sequence]
            if self.bitaverage:
                columns = [ str(zz) + '_drugAveraged' for zz in range(0, int(self.holder.shape[1]), 1)]
            a = copy.deepcopy(self.holder)
            a = pd.DataFrame(a, columns=columns, index=self.df2.index)
            if agg_duplicate_smiles:
                b = self.df2.loc[:,['cell_line_name','drug_row_col', x]]
            else:
                b = self.df2.loc[:,['cell_line_name', x]]
            out = pd.concat((b, a), axis=1)
            
            ready[x] = out
            
        ready['name'] = self.fps_name[36:]
        return ready
                                  