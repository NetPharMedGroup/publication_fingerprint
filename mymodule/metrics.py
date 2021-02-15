from scipy.stats import iqr, wilcoxon, normaltest, sem, t
from arch.bootstrap import IIDBootstrap
import numpy as np
import pandas as pd

def make_normalizer(df):
    
    normalizer = dict()
    normalizer['sd'] = dict(df.std(axis='index')) # get st.dev
    normalizer['iqr'] = dict(zip(normalizer['sd'].keys(), iqr(df, axis=0))) # get iqr
    normalizer['range'] = dict()
    for x in normalizer['sd'].keys(): # get range
        r = df[x].max() + abs(df[x].min())
        normalizer['range'][x] = r
    return normalizer

# calculate 95% CI using student t-distribution
# standardize rmse either by IQ25-75 or st.dev or range or none


def mean_confidence_interval_normed(data, metric, normalizer=None, bootstrap_reps=1000, norm='none', test_type='own', confidence=0.95): 
    '''
    norm can be st or iqr or range or none. 
    test_type: bs for BCa, t for t-stud, own for symmetric bootstrap, z for z-transform
    Normalizer should have all the data
    https://arch.readthedocs.io/en/latest/bootstrap/generated/generated/arch.bootstrap.IIDBootstrap.conf_int.html#arch.bootstrap.IIDBootstrap.conf_int
    '''
    if metric in ['css_ri', 'synergy_zip', 'synergy_bliss', 'synergy_hsa', 'synergy_loewe']:
        if norm in ['sd','iqr','range']:
            a = np.array(data)/normalizer[norm][metric]
            mean_val = np.mean(a)
        elif norm=='none':
            a = np.array(data)
            mean_val = np.mean(a)
        else:
            print('no norming info!')
            return
    if test_type == 't': # standard t-dist
        ci = _mean_confidence_interval(data=a, confidence=confidence)
    elif test_type == 'bs': # 
        # the idea is that we take mean to be as is, but we take its 95% CI bootstrapped
        n = len(a)
        # batch-correct and accelerated bootstrap. For now it defaults to 0.95. Fix by using partial 
        ci = IIDBootstrap(a).conf_int(_mean_confidence_interval, 
                                      reps=bootstrap_reps, 
                                      method='bca')
        ci = ci[1]
        mean_val = np.mean(a)
    elif test_type == 'own':
        '''(standard symmetrical bootstrap)'''
        mean_val, ci = _bootstrap(data=a, confidence=confidence, bootstrap_reps=bootstrap_reps)
    elif test_type == 'z':
        mean_val, ci = _pearsonr_ci(data=a, confidence=confidence)
        
    return round(mean_val, 4), round(float(ci), 4) # return upper bound

def _mean_confidence_interval(data=None, confidence=0.95):
    assert data is not None
    a = np.array(data)
    n = len(a)
    se = sem(a)
    h = se * t.ppf((1 + confidence) / 2, n-1)
    return h

def _bootstrap(data, confidence=0.95, bootstrap_reps=1000):
    """
    Generate n bootstrap samples, np.mean on each, sort means, find indexes of vals that correspond to the desired a levels
    """
    n = len(data)
    simulations = np.zeros((bootstrap_reps,),dtype=np.float32)
    xbar_init = np.mean(data)
    for index in range(bootstrap_reps):
        s = np.random.choice(data, size=n, replace=True)
        simulations[index] = np.mean(s)
    simulations = np.sort(simulations)

    u_pval = (1+confidence)/2.
    l_pval = (1-u_pval)
    l_indx = int(np.floor(bootstrap_reps*l_pval))
    u_indx = int(np.floor(bootstrap_reps*u_pval))
    return np.mean(data), (simulations[u_indx]-simulations[l_indx])

def _pearsonr_ci(data, confidence=0.95):
    ''' get conf interval doing z transform on CI values
    '''
    n = len(data)
    data_z = np.arctanh(data)
    se = sem(data_z)
    h = se * t.ppf((1 + confidence) / 2, n-1)
    return np.mean(data), np.tanh(h)

def make_confidence_data(bootstrap_reps=1000, test_type=None, log_file=None, normalizer_dict=None):
    '''
    test_type: bs for BCa, t for t-stud, own for symmetric bootstrap, z for z-transform
    log_file should have data read from teh disk
    normalizer dict should have drugcomb-wide vals with iqr,range,sd
    '''
    css={}
    zip_s={}
    loewe={}
    hsa={}
    bliss = {}
    for k,v in log_file.items():
        for kk,vv in v.items():
            if 'name' != kk:
                if kk == 'css_ri':
                    css[k] = vv
                elif kk == 'synergy_zip':
                    zip_s[k] = vv
                elif kk == 'synergy_bliss':
                    bliss[k] = vv
                elif kk == 'synergy_loewe':
                    loewe[k] = vv
                elif kk == 'synergy_hsa':
                    hsa[k] =vv
    # pearson r's
    css_ci95={}
    zip_ci95={}
    loewe_ci95={}
    hsa_ci95={}
    bliss_ci95 ={} # use defaultdict instead
    css_ci95['r'] = dict()
    zip_ci95['r'] = dict() 
    loewe_ci95['r'] = dict()
    hsa_ci95['r'] = dict() 
    bliss_ci95['r'] = dict() 

    for k,v in css.items():
        css_ci95['r'][k] = mean_confidence_interval_normed(data=[x[0][0] for x in v[0] ], 
                                                           metric='css_ri',
                                                           normalizer=normalizer_dict,
                                                           bootstrap_reps=bootstrap_reps,
                                                           test_type = test_type,
                                                           norm='none')
    for k,v in zip_s.items():
        zip_ci95['r'][k] = mean_confidence_interval_normed(data=[x[0][0] for x in v[0] ], 
                                                           metric='synergy_zip',
                                                           normalizer=normalizer_dict,
                                                           bootstrap_reps=bootstrap_reps,
                                                           test_type = test_type, 
                                                           norm='none' )
    for k,v in loewe.items():
        loewe_ci95['r'][k] = mean_confidence_interval_normed(data=[x[0][0] for x in v[0] ], 
                                                             metric='synergy_loewe',
                                                             normalizer=normalizer_dict,
                                                             bootstrap_reps=bootstrap_reps,
                                                             test_type = test_type, 
                                                             norm='none' )
    for k,v in bliss.items():
        bliss_ci95['r'][k] = mean_confidence_interval_normed(data=[x[0][0] for x in v[0] ], 
                                                             metric='synergy_bliss',
                                                             normalizer=normalizer_dict,
                                                             bootstrap_reps=bootstrap_reps,
                                                             test_type = test_type, 
                                                             norm='none' )
    for k,v in hsa.items():
        hsa_ci95['r'][k] = mean_confidence_interval_normed(data=[x[0][0] for x in v[0] ], 
                                                           metric='synergy_hsa',
                                                           normalizer=normalizer_dict,
                                                           bootstrap_reps=bootstrap_reps,
                                                           test_type = test_type, 
                                                           norm='none' )

    # add rmse's normed by st.dev
    css_ci95['sd'] = dict()
    zip_ci95['sd'] = dict() 
    loewe_ci95['sd'] = dict()
    hsa_ci95['sd'] = dict() 
    bliss_ci95['sd'] = dict() 

    for k,v in css.items():
        css_ci95['sd'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                            metric='css_ri',
                                                            normalizer=normalizer_dict,
                                                            bootstrap_reps=bootstrap_reps,
                                                            test_type = test_type, 
                                                            norm='sd' )
    for k,v in zip_s.items():
        zip_ci95['sd'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                            metric='synergy_zip',
                                                            normalizer=normalizer_dict,
                                                            bootstrap_reps=bootstrap_reps,
                                                            test_type = test_type, 
                                                            norm='sd' )
    for k,v in loewe.items():
        loewe_ci95['sd'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                              metric='synergy_loewe',
                                                              normalizer=normalizer_dict,
                                                              bootstrap_reps=bootstrap_reps,
                                                              test_type = test_type, 
                                                              norm='sd' )
    for k,v in bliss.items():
        bliss_ci95['sd'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                              metric='synergy_bliss',
                                                              normalizer=normalizer_dict,
                                                              bootstrap_reps=bootstrap_reps,
                                                              test_type = test_type, 
                                                              norm='sd' )
    for k,v in hsa.items():
        hsa_ci95['sd'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                            metric='synergy_hsa',
                                                            normalizer=normalizer_dict,
                                                            bootstrap_reps=bootstrap_reps,
                                                            test_type = test_type, 
                                                            norm='sd')

    # add rmse's normed by st.dev
    css_ci95['iqr'] = dict()
    zip_ci95['iqr'] = dict() 
    loewe_ci95['iqr'] = dict()
    hsa_ci95['iqr'] = dict() 
    bliss_ci95['iqr'] = dict() 

    for k,v in css.items():
        css_ci95['iqr'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                             metric='css_ri',
                                                             normalizer=normalizer_dict,
                                                             bootstrap_reps=bootstrap_reps,
                                                             test_type = test_type, 
                                                             norm='iqr' )
    for k,v in zip_s.items():
        zip_ci95['iqr'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                             metric='synergy_zip',
                                                             normalizer=normalizer_dict,
                                                             test_type = test_type, 
                                                             norm='iqr' )
    for k,v in loewe.items():
        loewe_ci95['iqr'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                               metric='synergy_loewe',
                                                               normalizer=normalizer_dict,
                                                               bootstrap_reps=bootstrap_reps,
                                                               test_type = test_type, 
                                                               norm='iqr' )
    for k,v in bliss.items():
        bliss_ci95['iqr'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                               metric='synergy_bliss',
                                                               normalizer=normalizer_dict,
                                                               bootstrap_reps=bootstrap_reps,
                                                               test_type = test_type, 
                                                               norm='iqr' )
    for k,v in hsa.items():
        hsa_ci95['iqr'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                             metric='synergy_hsa',
                                                             normalizer=normalizer_dict,
                                                             bootstrap_reps=bootstrap_reps,
                                                             test_type = test_type, 
                                                             norm='iqr')

    # normed by range
    css_ci95['range'] = dict()
    zip_ci95['range'] = dict() 
    loewe_ci95['range'] = dict()
    hsa_ci95['range'] = dict() 
    bliss_ci95['range'] = dict() 

    for k,v in css.items():
        css_ci95['range'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                               metric='css_ri',
                                                               normalizer=normalizer_dict,
                                                               bootstrap_reps=bootstrap_reps,
                                                               test_type = test_type, 
                                                               norm='range' )
    for k,v in zip_s.items():
        zip_ci95['range'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                               metric='synergy_zip',
                                                               normalizer=normalizer_dict,
                                                               bootstrap_reps=bootstrap_reps,
                                                               test_type = test_type, 
                                                               norm='range' )
    for k,v in loewe.items():
        loewe_ci95['range'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                                 metric='synergy_loewe',
                                                                 normalizer=normalizer_dict,
                                                                 bootstrap_reps=bootstrap_reps,
                                                                 test_type = test_type, 
                                                                 norm='range' )

    for k,v in bliss.items():
        bliss_ci95['range'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                                 metric='synergy_bliss',
                                                                 normalizer=normalizer_dict,
                                                                 bootstrap_reps=bootstrap_reps,
                                                                 test_type = test_type, 
                                                                 norm='range' )

    for k,v in hsa.items():
        hsa_ci95['range'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                               metric='synergy_hsa',
                                                               normalizer=normalizer_dict,
                                                               bootstrap_reps=bootstrap_reps,
                                                               test_type = test_type, 
                                                               norm='range') 
    # normed by range
    css_ci95['no_norm'] = dict()
    zip_ci95['no_norm'] = dict() 
    loewe_ci95['no_norm'] = dict()
    hsa_ci95['no_norm'] = dict() 
    bliss_ci95['no_norm'] = dict() 

    for k,v in css.items():
        css_ci95['no_norm'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                                 metric='css_ri',
                                                                 normalizer=normalizer_dict,
                                                                 bootstrap_reps=bootstrap_reps,
                                                                 test_type = test_type, 
                                                                 norm='none' )
    for k,v in zip_s.items():
        zip_ci95['no_norm'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ],
                                                                 metric='synergy_zip',
                                                                 normalizer=normalizer_dict,
                                                                 bootstrap_reps=bootstrap_reps,
                                                                 test_type = test_type, 
                                                                 norm='none' )
    for k,v in loewe.items():
        loewe_ci95['no_norm'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ],
                                                                   metric='synergy_loewe',
                                                                   normalizer=normalizer_dict,
                                                                   bootstrap_reps=bootstrap_reps,
                                                                   norm='none' ,
                                                                   test_type = test_type)
    for k,v in bliss.items():
        bliss_ci95['no_norm'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ],
                                                                   metric='synergy_bliss',
                                                                   normalizer=normalizer_dict,
                                                                   bootstrap_reps=bootstrap_reps,
                                                                   norm='none',
                                                                   test_type = test_type )
    for k,v in hsa.items():
        hsa_ci95['no_norm'][k] = mean_confidence_interval_normed(data=[x[1] for x in v[0] ], 
                                                                 metric='synergy_hsa', 
                                                                 normalizer=normalizer_dict,
                                                                 bootstrap_reps=bootstrap_reps,
                                                                 norm='none',
                                                                 test_type = test_type) 
    return css_ci95,zip_ci95,loewe_ci95,bliss_ci95,hsa_ci95

