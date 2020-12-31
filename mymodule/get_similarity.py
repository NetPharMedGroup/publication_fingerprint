import numpy as np
import pickle
import pandas as pd
from time import time
import torch
from . import Trasnformer_model
from . import VAE_model
from . import GAE_model
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.autograd import Variable
import warnings
from scipy.linalg import svd

# https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb

processor = VAE_model.ProcessSMILES()


def get_similarity(n_samples=100, smiles_set=None, VAE=16, TRANSFORMER=64, positive_control='VAE', slice_G=0, slice_T=0, seed=None,step_size=10, unbiased_gram=True):
    '''
    vae/transformer: for now either VAE or T
    
    vocab: for transformer
    smiles_set: should be b, usually (ie all found drugcomb) 
    '''
    if smiles_set is None:
        with open('/tf/notebooks/code_for_pub/smiles_files/smiles_drugcomb_BY_cid_duplicated.pickle','rb') as f:
            b = pickle.load(f)
            smiles_set = b.drop_duplicates().reset_index(drop=True)
    if seed is None:
        idx = np.random.choice(range(len(smiles_set)), size=n_samples, replace=False) 
        idx1 = np.random.choice(range(len(smiles_set)), size=n_samples, replace=False) 
        n_overlap = len([x for x in idx if x in idx1])
    else:
        rs = np.random.RandomState(seed)
        idx = rs.choice(range(len(smiles_set)), size=n_samples, replace=False) 
        rs1 = np.random.RandomState(seed+1)
        idx1 = rs1.choice(range(len(smiles_set)), size=n_samples, replace=False) 
        n_overlap = len([x for x in idx if x in idx1])
    # slice VAE
    if slice_T == 0:
        start = 0
        end = VAE
        name = 'mean'
    elif slice_T == 1:
        start = VAE
        end = VAE+VAE
        name = 'max'
    elif slice_T == 2:
        start = VAE+VAE
        end = VAE+VAE+VAE
        name = 'first output of last layer'
    elif slice_T == 3:
        start = VAE+VAE+VAE
        end = VAE+VAE+VAE+VAE
        name = 'first output of second last layer'
    elif slice_T == 'all':
        start = 0
        end = VAE+VAE+VAE+VAE
        name = 'all'
    # slice GAE
    if slice_G == 0:
        start_G = 0
        end_G = 16
        name_G = 'mean'
    elif slice_G == 1:
        start_G = 16
        end_G = 32
        name_G = 'min'
    elif slice_G == 2:
        start_G = 32
        end_G = 48
        name_G = 'max'
    elif slice_G == 3:
        start_G = 48
        end_G = 64
        name_G = 'svd'
    elif slice_G == 'all':
        start_G = 0
        end_G = 64
        name_G = 'all'
        
    ##############
    ## prep GAE ##
    ##############
    start_t = time()
    PATH = '/tf/notebooks/code_for_pub/model_files/GAE_model_best'
    device = torch.device('cpu')
    m = torch.load(PATH, map_location=device)
    model = GAE_model.GAE(in_dim=54, hidden_dims=[54,46,40,34,28,22,16])
    model.load_state_dict(m)
    model.eval()
    #####################
    ## prep GAE loader ##
    #####################
    #1
    mols = GAE_model.smiles2mols(smiles_set.iloc[idx])
    graphs = GAE_model.mols2graphs(mols)
    d_GAE=[]
    for i in graphs:
        a = model.encode(i).detach().numpy()
        d_GAE.append(a)
    #2
    mols1 = GAE_model.smiles2mols(smiles_set.iloc[idx1])
    graphs1 = GAE_model.mols2graphs(mols1)
    d_GAE1=[]
    for i in graphs1:
        a = model.encode(i).detach().numpy()
        d_GAE1.append(a)
    ##################
    ## make GAE fps ##
    ##################
    gae_fps = np.zeros((n_samples, 64), dtype=np.float)
    gae_fps1 = np.zeros((n_samples, 64), dtype=np.float)
    for index,i in enumerate(d_GAE):
        me = np.mean(i, axis=0) # aggregation over representation
        mi = np.min(i, axis=0)
        ma = np.max(i, axis=0)
        sv = svd(i, compute_uv=False)
        if sv.shape[0] != 16: 
            # zero padding, in case of small molecules
            result = np.zeros((16,))
            result[:sv.shape[0]] = sv
            sv = result.copy()
        fp = np.append([me,mi,ma],sv)
        gae_fps[index] = fp
    for index,i in enumerate(d_GAE1):
        me = np.mean(i, axis=0) # aggregation over representation
        mi = np.min(i, axis=0)
        ma = np.max(i, axis=0)
        sv = svd(i, compute_uv=False)
        if sv.shape[0] != 16: 
            # zero padding, in case of small molecules
            result = np.zeros((16,))
            result[:sv.shape[0]] = sv
            sv = result.copy()
        fp = np.append([me,mi,ma],sv)
        gae_fps1[index] = fp
    print(f'made GAE in {round((time()-start_t),3)}')
    ##############
    ## prep VAE ##
    ##############
    start_t = time()
    if VAE == 16:
        # define model
        m = torch.load('/tf/notebooks/code_for_pub/model_files/VAE16_model_best.pth.tar')
        encoder = VAE_model.MolEncoder(i=140, o=16, c=len(m['charset']))
        encoder.load_state_dict(m['encoder'])
        encoder.eval()
    elif VAE == 256:
        # define model
        m = torch.load('/tf/notebooks/code_for_pub/model_files/VAE256_model_best.pth.tar')
        encoder = VAE_model.MolEncoder(i=140, o=256, c=len(m['charset']))
        encoder.load_state_dict(m['encoder'])
        encoder.eval()
    #####################
    ## prep VAE loader ##
    #####################
    ## 1
    temp = []
    for i in smiles_set.iloc[idx]:
        temp.append(processor.one_hot_encode(STRING=i, charset=m['charset'], size_of_encoding=len(m['charset']), max_len_smiles=140, dtype=np.uint8))
    temp = torch.from_numpy(np.array(temp, dtype=np.uint8))
    vae = TensorDataset(temp, torch.zeros(temp.size()[0]))
    v_loader = DataLoader(vae, batch_size=1, shuffle=False)
    ## 2
    temp1 = []
    for i in smiles_set.iloc[idx1]:
        temp1.append(processor.one_hot_encode(STRING=i, charset=m['charset'], size_of_encoding=len(m['charset']), max_len_smiles=140, dtype=np.uint8))
    temp1 = torch.from_numpy(np.array(temp1, dtype=np.uint8))
    vae1 = TensorDataset(temp1, torch.zeros(temp1.size()[0]))
    v_loader1 = DataLoader(vae1, batch_size=1, shuffle=False)
    ##################
    ## make VAE fps ##
    ##################
    vae_fps = np.zeros((n_samples, VAE), dtype=np.float)
    vae_fps1 = np.zeros((n_samples, VAE), dtype=np.float)
    with torch.no_grad():
        for ind, (x,_) in enumerate(v_loader):
            x_var = Variable(x.type(torch.FloatTensor))
            vae_fps[ind] = encoder(x_var).detach().numpy()
    with torch.no_grad():
        for ind, (x,_) in enumerate(v_loader1):
            x_var = Variable(x.type(torch.FloatTensor))
            vae_fps1[ind] = encoder(x_var).detach().numpy()
    print(f'made VAE in {round((time()-start_t),3)}')
    ######################
    ## prep Trasnformer ##
    ######################
    start_t = time()
    if TRANSFORMER == 64:
        m = torch.load('/tf/notebooks/code_for_pub/model_files/T16_model_best.pth.tar')
        t = Trasnformer_model.TrfmSeq2seq(in_size=len(m['vocab']), hidden_size = 16, out_size=len(m['vocab']), n_layers=6)
        t.load_state_dict(m['state_dict'])
        t.eval()
    if TRANSFORMER == 1024:
        m = torch.load('/tf/notebooks/code_for_pub/model_files/T256_model_best.pth.tar')
        t = Trasnformer_model.TrfmSeq2seq(in_size=len(m['vocab']), hidden_size = 256, out_size=len(m['vocab']), n_layers=6)
        t.load_state_dict(m['state_dict'])
        t.eval()
    #############################
    ## prep Trasnformer loader ##
    #############################    
    ## 1
    dataset = Trasnformer_model.Seq2seqDataset(smiles_set[idx], m['vocab'], seq_len=145)
    t_loader = DataLoader(Subset(dataset,idx), batch_size=1, shuffle=False, num_workers=1)
    ## 2
    dataset1 = Trasnformer_model.Seq2seqDataset(smiles_set[idx1], m['vocab'], seq_len=145)
    t_loader1 = DataLoader(Subset(dataset1, idx1), batch_size=1, shuffle=False, num_workers=1)
    ##########################
    ## make Transformer fps ##
    ##########################
    transformer_fps = np.zeros((n_samples, TRANSFORMER), dtype=np.float)
    transformer_fps1 = np.zeros((n_samples, TRANSFORMER), dtype=np.float)
    with torch.no_grad():
        for ind, sm in enumerate(t_loader):
            transformer_fps[ind]= t.encode(torch.t(sm)).reshape(-1)
    with torch.no_grad():
        for ind, sm in enumerate(t_loader1):
            transformer_fps1[ind]= t.encode(torch.t(sm)).reshape(-1)
    print(f'made T in {round((time()-start_t),3)}')
    
    ###
    # calculate similarity
    if n_samples > 300:
        counter = list(range(0,n_samples,step_size))
    else:
        counter = list(range(n_samples))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # same fps, different models
        holder_sameData_difModels_tv = []
        holder_sameData_difModels_tg = []
        holder_sameData_difModels_vg = []
        start_t = time()
        for x in counter:
            x1 = transformer_fps[:x,start:end]
            y1 = vae_fps[:x,:]
            z1 = gae_fps[:x,start_G:end_G]
            try:
                cka_from_examples_debiased_tv = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram) # t vs v
                holder_sameData_difModels_tv.append(cka_from_examples_debiased_tv)
            except: continue
            try:
                cka_from_examples_debiased_tg = cka(gram_linear(x1), gram_linear(z1), debiased=unbiased_gram) # t vs g
                holder_sameData_difModels_tg.append(cka_from_examples_debiased_tg)
            except: continue    
            try:
                cka_from_examples_debiased_vg = cka(gram_linear(y1), gram_linear(z1), debiased=unbiased_gram) # v vs g
                holder_sameData_difModels_vg.append(cka_from_examples_debiased_vg)
            except:
                continue
        holder_sameData_difModels_tv = [x for x in holder_sameData_difModels_tv if not np.isnan(x)]
        holder_sameData_difModels_tg = [x for x in holder_sameData_difModels_tg if not np.isnan(x)]
        holder_sameData_difModels_vg = [x for x in holder_sameData_difModels_vg if not np.isnan(x)]
        
        holder_sameData_difModels_tv1 = []
        holder_sameData_difModels_tg1 = []
        holder_sameData_difModels_vg1 = []
        for x in counter:
            try:
                # same. dif models
                x2 = transformer_fps1[:x,start:end]
                y2 = vae_fps1[:x,:]
                z2 = gae_fps1[:x,start_G:end_G]
                cka_from_examples_debiased_tv1 = cka(gram_linear(x2), gram_linear(y2), debiased=unbiased_gram) # t vs v
                holder_sameData_difModels_tv1.append(cka_from_examples_debiased_tv1)
                
                cka_from_examples_debiased_tg1 = cka(gram_linear(x2), gram_linear(z2), debiased=unbiased_gram) # t vs g
                holder_sameData_difModels_tg1.append(cka_from_examples_debiased_tg1)
                
                cka_from_examples_debiased_vg1 = cka(gram_linear(y2), gram_linear(z2), debiased=unbiased_gram) # v vs g
                holder_sameData_difModels_vg1.append(cka_from_examples_debiased_vg1)
            except:
                continue
        holder_sameData_difModels_tv1 = [x for x in holder_sameData_difModels_tv1 if not np.isnan(x)]
        holder_sameData_difModels_tg1 = [x for x in holder_sameData_difModels_tg1 if not np.isnan(x)]
        holder_sameData_difModels_vg1 = [x for x in holder_sameData_difModels_vg1 if not np.isnan(x)]

        print(f'same fps, diff models in {round((time()-start_t),3)}')
        # different fps, same models 
        #1
        start_t = time()
        holder_difData_sameModels_v = []
        for x in counter:
            try:
                x1 = vae_fps[:x,:]
                y1 = vae_fps1[:x,:]
                cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                holder_difData_sameModels_v.append(cka_from_examples_debiased)
            except:
                continue
        holder_difData_sameModels_v = [x for x in holder_difData_sameModels_v if not np.isnan(x)]
        print(f'diff fps, same models ver1 in {round((time()-start_t),3)}')
        #2
        holder_difData_sameModels_t = []
        start_t = time()
        for x in counter:
            try:
                x1 = transformer_fps[:x,start:end]
                y1 = transformer_fps1[:x,start:end]
                cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                holder_difData_sameModels_t.append(cka_from_examples_debiased)
            except:
                continue
        holder_difData_sameModels_t = [x for x in holder_difData_sameModels_t if not np.isnan(x)]
        print(f'diff fps, same models ver2 in {round((time()-start_t),3)}')
        #3
        holder_difData_sameModels_g = []
        start_t = time()

        for x in counter:
            try:
                x1 = gae_fps[:x,start_G:end_G]
                y1 = gae_fps1[:x,start_G:end_G]
                cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                holder_difData_sameModels_g.append(cka_from_examples_debiased)
            except:
                continue
        holder_difData_sameModels_g = [x for x in holder_difData_sameModels_g if not np.isnan(x)]
        print(f'diff fps, same models ver2 in {round((time()-start_t),3)}')
        # positive control. Same fps, same model
        start_t = time()
        if positive_control=='VAE':
            positive_c = []
            for x in counter:
                try:
                    x1 = vae_fps[:x,:]
                    y1 = vae_fps[:x,:]
                    cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                    positive_c.append(cka_from_examples_debiased)
                except:
                    continue
            positive_c = [x for x in positive_c if not np.isnan(x)]
        elif positive_control=='Transformer':
            positive_c = []
            for x in counter:
                try:
                    x1 = transformer_fps[:x,start:end]
                    y1 = transformer_fps[:x,start:end]
                    cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                    positive_c.append(cka_from_examples_debiased)
                except:
                    continue
            positive_c = [x for x in positive_c if not np.isnan(x)]
        elif positive_control=='GAE':
            positive_c = []
            for x in counter:
                try:
                    x1 = gae_fps[:x,start_G:end_G]
                    y1 = gae_fps[:x,start_G:end_G]
                    cka_from_examples_debiased = cka(gram_linear(x1), gram_linear(y1), debiased=unbiased_gram)
                    positive_c.append(cka_from_examples_debiased)
                except:
                    continue
            positive_c = [x for x in positive_c if not np.isnan(x)]
        print(f'positive control in {round((time()-start_t), 3)}')
        # negative control. two random arrays
        negative_control = []
        start_t = time()
        r = np.random.uniform(-5,5,(n_samples,VAE))
        r1 = np.random.uniform(-5,5,(n_samples,VAE))
        for x in counter:
            try:
                cka_from_examples_debiased = cka(gram_linear(r[:x]), gram_linear(r1[:x]), debiased=unbiased_gram)
                negative_control.append(cka_from_examples_debiased)
            except:
                continue
        negative_control = [x for x in negative_control if not np.isnan(x)]
        print(f'negative control in {round((time()-start_t), 3)}')

    f=plt.figure(figsize=(14,12))
    
    plt.scatter(range(len(holder_difData_sameModels_v)),holder_difData_sameModels_v, marker='.',c='#003366', 
                label=f'Diff drugs, same model VAE, drugs overlap={n_overlap}, avg={round(np.mean(holder_difData_sameModels_v),4)}')
    plt.scatter(range(len(holder_difData_sameModels_t)),holder_difData_sameModels_t, marker='.',c='#0080FF', 
                label=f'Diff drugs, same model Trans, drugs overlap={n_overlap}, avg={round(np.mean(holder_difData_sameModels_t),4)}')
    plt.scatter(range(len(holder_difData_sameModels_g)),holder_difData_sameModels_g, marker='.',c='#CCE5FF', 
                label=f'Diff drugs, same model GAE, drugs overlap={n_overlap}, avg={round(np.mean(holder_difData_sameModels_g),4)}')
    
    plt.scatter(range(len(holder_sameData_difModels_tg)),holder_sameData_difModels_tg, marker='.',c='#FFCCCC', 
                label=f'Same drugs, TvsG #1, avg={round(np.mean(holder_sameData_difModels_tg),4)}')
    plt.scatter(range(len(holder_sameData_difModels_tg1)),holder_sameData_difModels_tg1, marker='.',c='#FF9999', 
                label=f'Same drugs, TvsG #2, avg={round(np.mean(holder_sameData_difModels_tg1),4)}')
    
    plt.scatter(range(len(holder_sameData_difModels_tv)),holder_sameData_difModels_tv, marker='.',c='#FFFFCC', 
                label=f'Same drugs, TvsV #1, avg={round(np.mean(holder_sameData_difModels_tv),4)}')
    plt.scatter(range(len(holder_sameData_difModels_tv1)),holder_sameData_difModels_tv1, marker='.',c='#FFFF99', 
                label=f'Same drugs, TvsV #2, avg={round(np.mean(holder_sameData_difModels_tv1),4)}')
    
    plt.scatter(range(len(holder_sameData_difModels_vg)),holder_sameData_difModels_vg, marker='.',c='#E5CCFF', 
                label=f'Same drugs, VvsG #1, avg={round(np.mean(holder_sameData_difModels_vg),4)}')
    plt.scatter(range(len(holder_sameData_difModels_vg1)),holder_sameData_difModels_vg1, marker='.',c='#CC99FF', 
                label=f'Same drugs, VvsG #2, avg={round(np.mean(holder_sameData_difModels_vg1),4)}')
    
    plt.scatter(range(len(negative_control)),negative_control, marker='o',c='w',label=f'Negative control - random numbers')
    plt.scatter(range(len(positive_c)),positive_c, marker='.',c='k',label=f'Positive control using 2x{positive_control}')
    plt.ylabel('CKA similarity')
    plt.xlabel(f'CKA calculation performed every {step_size}')
    plt.ylim( (-1,1.05) )
    plt.title(f'VAE{VAE} vs Transformer{TRANSFORMER}-{name} vs GAE-{name_G} on {n_samples} samples')
    plt.legend(loc='lower right', prop={'size': 6})
    plt.show()
    return f
        
        
        

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)