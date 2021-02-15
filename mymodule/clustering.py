import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

def plot_single_projection(holder, 
                           labels, 
                           class_name='Antioxidants', 
                           fp_name ='fps_e3fp_1024bit', 
                           standardize=True, 
                           preprocess_lda='PCA'):
    '''
    holder should be a dictionary with df's as values and fp-filenames as keys
    labels should be a mapping of DrugCombID: ATC_class
    '''

    from mlxtend.preprocessing import standardize as st
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from mlxtend.feature_extraction import LinearDiscriminantAnalysis #in sklearn LDA i'd need to add a dummy class if i want to have 2 components after trasnformation
    from scipy.spatial.distance import pdist

    df_cluster = holder[fp_name].copy()
    df_cluster = df_cluster.loc[df_cluster.index.isin(labels.keys())]
    df_cluster = df_cluster[~df_cluster.index.duplicated(keep='last')]

    if standardize:
        classes = df_cluster.index.copy()
        df_cluster.reset_index(inplace=True, drop=True)
        df_cluster = st(df_cluster)
    else:
        classes = df_cluster.index.copy()
    df_cluster['classes'] = classes # our classes are mapped to index in labels dictionary
    df_cluster['classes'] = df_cluster['classes'].map(labels)

    df_cluster.loc[df_cluster.classes != class_name,'classes'] = 'not '+'class_name'
    #dummy = [0]*(df_cluster.shape[1]-1) + ['dummy']
    #df_cluster.loc[df_cluster.shape[0]] = dummy

    # change labels from str to int
    enc = LabelEncoder() 
    real_classes = df_cluster.loc[:,'classes']
    df_cluster.loc[:,'classes'] = enc.fit_transform(df_cluster['classes'])
    classes = df_cluster.pop('classes')

    if preprocess_lda == 'PLS':
        from sklearn.cross_decomposition import PLSRegression
        pls = PLSRegression(n_components=10, scale=False)
        temp = pls.fit_transform(df_cluster.values, classes.values)[0]
    elif preprocess_lda == 'PCA':
        from sklearn.decomposition import PCA 
        pca = PCA(n_components=0.95, svd_solver='full', whiten=False)
        temp = pca.fit_transform(df_cluster.values)
    elif preprocess_lda == 'kernelPCA':
        from sklearn.decomposition import KernelPCA
        pca = KernelPCA(kernel="rbf", gamma=5)
        temp = pca.fit_transform(df_cluster.values)
    elif preprocess_lda == 'NONE':
        temp = df_cluster.values
    elif preprocess_lda == 'NCA':
        from sklearn.neighbors import NeighborhoodComponentsAnalysis
        nca = NeighborhoodComponentsAnalysis()
        temp = nca.fit_transform(df_cluster.values, classes.values)
            

    #lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    #lda.fit(temp, classes.values)
    #temp1 = lda.transform(temp)

    lda = LinearDiscriminantAnalysis(n_discriminants=2)
    lda.fit(temp, classes.values)
    temp = lda.transform(temp)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')
        temp = temp.astype(np.float) # in case of complex numbers///

    df = pd.DataFrame(index=df_cluster.index, columns=[0,1], data=temp)
    df['classes'] = real_classes


    km = KMeans(init='k-means++', n_clusters=1, n_init=10)
    km.fit(df.loc[df.classes != class_name, [0,1]])

    km1 = KMeans(init='k-means++', n_clusters=1, n_init=10)
    km1.fit(df.loc[df.classes == class_name, [0,1]])

    d = pdist([km.cluster_centers_[0], km1.cluster_centers_[0]])
    d = str(round(d[0],3))


    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(df.loc[df.classes != class_name, 0],
                 df.loc[df.classes != class_name, 1],
                 marker=',', 
                 color='grey')
    ax.scatter(df.loc[df.classes == class_name, 0],
                 df.loc[df.classes == class_name, 1],
                 marker=',', 
                 color='orange')

    ax.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], marker='X', color='green', linewidths=30)

    ax.scatter(km1.cluster_centers_[:,0], km1.cluster_centers_[:,1], marker='X', color='red', linewidths=30)

    fig.suptitle(class_name + ' ' + d)
    return fig

def plot_projections(holder, 
                     labels, 
                     preprocess_lda='PCA', 
                     class_name='Antioxidants', 
                     only_pca=False, 
                     binarize_class=True,
                     standardize=True,
                     cluster=True,
                     return_distances=False
                    ):
    '''
    holder should be a dictionary with df's as values and fp-filenames as keys
    labels should be a mapping of DrugCombID: ATC_class
    '''
    if only_pca:
        from sklearn.decomposition import PCA
        
        df = dict()
        for ind, i in enumerate(['fps_e3fp_1024bit', 'fps_morgan_1024bit','fps_topo_1024bit','fps_infomax_new',
                  'fps_VAE_256bit_new','fps_VAE_16bit_new','fps_transformer_1024bit_new',
                  'fps_transformer_64bit_new','fps_gae_64bit_new']):

            df_cluster = holder[i].copy()
            df_cluster = df_cluster.loc[df_cluster.index.isin(labels.keys())]
            df_cluster = df_cluster[~df_cluster.index.duplicated(keep='last')]
            if standardize:
                from mlxtend.preprocessing import standardize as st
                classes = df_cluster.index.copy()
                df_cluster.reset_index(inplace=True, drop=True)
                df_cluster = st(df_cluster)
            else:
                classes = df_cluster.index.copy()
            pca = PCA(n_components=2)
            temp = pca.fit_transform(df_cluster)
            df[ind] = pd.DataFrame(index=df_cluster.index, data=temp)
            df[ind]['classes'] = classes
            df[ind]['classes'] = df[ind]['classes'].map(labels)
        title = 'PCA'
        
    else: # to LDA
        from mlxtend.feature_extraction import LinearDiscriminantAnalysis as LDA
        from sklearn.preprocessing import LabelEncoder
        # binary https://stats.stackexchange.com/questions/178587/why-is-the-rank-of-covariance-matrix-at-most-n-1/180366#180366

        df = dict()
        for ind, i in enumerate(['fps_e3fp_1024bit', 'fps_morgan_1024bit','fps_topo_1024bit','fps_infomax_new',
                  'fps_VAE_256bit_new','fps_VAE_16bit_new','fps_transformer_1024bit_new',
                  'fps_transformer_64bit_new','fps_gae_64bit_new']):
            
            df_cluster = holder[i].copy()
            df_cluster = df_cluster.loc[df_cluster.index.isin(labels.keys())]
            df_cluster = df_cluster[~df_cluster.index.duplicated(keep='last')]
            if standardize:
                from mlxtend.preprocessing import standardize as st
                from sklearn.preprocessing import MinMaxScaler

                classes = df_cluster.index.copy()
                df_cluster.reset_index(inplace=True, drop=True)
                mms = MinMaxScaler()
                df_cluster = pd.DataFrame(data = mms.fit_transform(df_cluster), 
                                          index = df_cluster.index, 
                                          columns=df.columns)
            else:
                classes = df_cluster.index.copy()
            df_cluster['classes'] = classes
            df_cluster['classes'] = df_cluster['classes'].map(labels)
            if binarize_class:
                df_cluster.loc[df_cluster.classes != class_name,'classes'] = 'not '+'class_name'
           
            # change labels from str to int
            enc = LabelEncoder() 
            real_classes = df_cluster.loc[:,'classes']
            df_cluster.loc[:,'classes'] = enc.fit_transform(df_cluster['classes'])
            classes = df_cluster.pop('classes')
            
            if preprocess_lda == 'PLS':
                from sklearn.cross_decomposition import PLSRegression
                pls = PLSRegression(n_components=10, scale=False)
                temp = pls.fit_transform(df_cluster.values, classes.values)[0]
            elif preprocess_lda == 'PCA':
                from sklearn.decomposition import PCA 
                pca = PCA(n_components=0.95, svd_solver='full', whiten=False)
                temp = pca.fit_transform(df_cluster.values)
            elif preprocess_lda == 'kernelPCA':
                from sklearn.decomposition import KernelPCA
                pca = KernelPCA(kernel="rbf", gamma=5)
                temp = pca.fit_transform(df_cluster.values)
            elif preprocess_lda == 'NONE':
                temp = df_cluster.values
            
            # lda
            lda = LDA(n_discriminants=2)
            lda.fit(temp, classes.values)
            temp = lda.transform(temp)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', 'Casting complex values to real discards the imaginary part')
                temp = temp.astype(np.float) # in case of complex numbers///
            df[ind] = pd.DataFrame(index=df_cluster.index, columns=[0,1], data=temp)
            df[ind]['classes']  = real_classes
          

        title = 'LDA'
        
    sns.set_context(context='talk')
    sns.set_style('dark')
    sns.set_style({'font.family':'serif', 'font.sans-serif':['Helvetica']})
    fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, figsize=(13,14))
    cm = plt.cm.get_cmap('Spectral')
    my_cmap = cm(np.linspace(0, 1, len( np.unique(df[ind]['classes']) )), alpha=0.6)
    
    if return_distances:
        distances = dict()
        sil_scores = dict()
        chs_scores = dict()
    for ax_n, key, x, name in zip([ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9], 
                             df.keys(),
                             df.values(),
                             ['E3FP','Morgan_300','Topo_1024','Infomax','VAE_256','VAE_16','Trans_1024',
                              'Trans_64','GAE_64']):
        if not binarize_class:
            for ind, i in enumerate(np.unique(x['classes'])):
                color=my_cmap[ind]
                marker='.'
                if i==class_name:
                    color='black',
                    marker=','
                ax_n.scatter(x.loc[x.classes==i, 0],
                             x.loc[x.classes==i, 1],
                             marker=marker, 
                             label=i + f' (n={str(len(x.loc[x.classes==i, 0]))}) vs Rest ({str(len(x.loc[x.classes!=i, 0]))})',
                             color=color)
                ax_n.title.set_text(name)
        else:
            ax_n.scatter(x.loc[:,0], x.loc[:,1], marker='.')
            ax_n.scatter(x.loc[x.classes==class_name, 0],
                         x.loc[x.classes==class_name, 1],
                         marker=',', 
                         label=class_name + f' (n={str(len(x.loc[x.classes==class_name, 0]))}) vs rest (n={str(len(x.loc[x.classes!=class_name, 0]))})',
                         color='darkorange')
            ax_n.title.set_text(name)
            if cluster:
                from sklearn.cluster import KMeans
                from scipy.spatial.distance import pdist
                from sklearn.metrics import silhouette_score as sil
                from sklearn.metrics import calinski_harabasz_score as chs
                
                km = KMeans(init='k-means++', n_clusters=1, n_init=10)
                km.fit(x.loc[x.classes != class_name, [0,1]])
                
                km1 = KMeans(init='k-means++', n_clusters=1, n_init=10)
                km1.fit(x.loc[x.classes == class_name, [0,1]])

                ax_n.scatter(km.cluster_centers_[:,0], 
                             km.cluster_centers_[:,1], 
                             marker='X', 
                             color='darkblue', 
                             s=100,
                             linewidth = 3
                            )
                ax_n.scatter(km1.cluster_centers_[:,0], 
                             km1.cluster_centers_[:,1], 
                             marker='X', 
                             color='red', 
                             s=100,
                             linewidth = 3)
                
                d = round(pdist([km.cluster_centers_[0], km1.cluster_centers_[0]], metric='euclidean')[0], 3)
                d_sc = round(sil(x.loc[:, [0,1]], x['classes']), 3)
                d_chs = round(chs(x.loc[:, [0,1]], x['classes']), 3)
                if return_distances:
                    cl_name = class_name +' '+name
                    distances[cl_name] = d
                    sil_scores[cl_name] = d_sc
                    chs_scores[cl_name] = d_chs
                name = name + '\n|d:'+str(d)+'|sil:'+str(d_sc)+'|chs:'+str(d_chs)
                ax_n.title.set_text(name)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_xticks([])
        ax.set_yticks([])

    labels = ax_n.get_legend_handles_labels()[1]
    if only_pca:
        fig.suptitle(labels[0] + "\n classified with: " + title)
    else:
        fig.suptitle(labels[0] + "\n classified with: " + title + f', preprocessed with: {preprocess_lda}')
    fig.tight_layout()
    if not return_distances:
        return fig
    else:
        return fig, distances, sil_scores, chs_scores


def get_bulk_similarity(df, n_drugs=10, cosine=False, scale=True):
    '''
    calculates tanimoto or cosine similarity. 
    If scale=True, first zero mean/one var, then minmax to {0,1}
    then calc similarity
    '''    
    similarity = []
    if not cosine:
        from rdkit.DataStructs.cDataStructs import CreateFromBitString as cfbt
        from rdkit.DataStructs import BulkTanimotoSimilarity as bts
        fps = []
        for x in range(n_drugs):
            fps.append(df.iloc[x,0:].tolist())
        n_fps = len(fps)
        
        bit_fps = [cfbt(''.join([str(x) for x in fp])) for fp in fps] #make sparse vectors
        for i in range(n_fps):
            sims = bts(bit_fps[i], bit_fps[i:]) # bulk tanimoto
            similarity.extend(sims)
    else:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        from scipy.spatial.distance import cosine #cosine difference, so similariy 1-diff
        # scaling
        if scale:
            scaler = StandardScaler()
            out = scaler.fit_transform(df) # scale zero mean, one var
            scaler = MinMaxScaler()
            out = scaler.fit_transform(out) # fit between 0 and 1
        else:
            out = df.values
        # getting fps
        fps = np.zeros((n_drugs, out.shape[1]), dtype=np.float) # make array to hold scaled fingerprints
        for ind, x in enumerate(range(n_drugs)):
            fps[ind,:] = out[x,:]
        
        # calc similarity
        for i in range(n_drugs):
            sims = [1-cosine(fps[i,:], fps[n,:]) for n in range(i, n_drugs, 1)]
            similarity.extend(sims)
    
    lower = np.zeros((n_drugs, n_drugs))
    lower[np.triu_indices(n_drugs)] = np.round(similarity, 3)
    lower[np.tril_indices(n_drugs,-1)] = np.NaN
    
    return pd.DataFrame(data=lower, index=df.index[:n_drugs], columns=df.index[:n_drugs])
