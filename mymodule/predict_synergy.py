import numpy as np
import pandas as pd
import time
import pickle
from collections import defaultdict
from progiter import ProgIter
from catboost import Pool, CatBoostRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def run_train(file, fp_name, cv=10, for_valid=0.4, ordered = False, ram_fraction=0.95, save=False, cv_params=None):
    cv_lower = 1
    cv_higher = 1+cv
    if cv_params is None:
        cv_params = dict()
        cv_params['bootstrap_type'] = 'Poisson'
        cv_params['l2_leaf_reg'] = 9
        cv_params['learning_rate'] = 0.15
        cv_params['depth'] = 10
        cv_params['cat_features'] = ['cell_line_name']
        cv_params['use_best_model'] = True
        cv_params['early_stopping_rounds'] = 50
        cv_params['iterations'] = 5000
        cv_params['task_type'] = 'GPU'
    else:
        cv_params = cv_params
    if ordered:
        cv_params['boosting_type'] = 'Ordered'
    
    cat_features = cv_params['cat_features']
    cv_params['gpu_ram_part'] = ram_fraction

    f = for_valid
    c= defaultdict(list)
    
    for k in ProgIter(['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa','css_ri','name'], total=5, verbose=1):
        v_temp = file[k]
        if k != 'name':
            if 'drug_row_col' in v_temp.columns:
                v = v_temp.drop(columns=['drug_row_col'], inplace=False)
            else:
                v = v_temp
            size = int(v.shape[0]*f) # 40% for valid
            a = []
            for i in range(cv_lower,cv_higher,1):
                print(k)
                # sampling
                np.random.seed(i)
                idx_valid = pd.Index(np.random.choice(v.index, size, replace=False))
                idx_test = v.index.difference(idx_valid)
                train = v.loc[idx_test,:] # returns df without the dropped idx 
                valid = v.loc[idx_valid,:]

                #prep datasets
                true_labels = valid.pop(k)
                y = train.pop(k)
                eval_dataset=Pool(valid, true_labels, cat_features=cat_features)

                #create a model
                model = CatBoostRegressor(**cv_params)
                model.fit(train,y, eval_set=eval_dataset, plot=False, verbose=1000)

                # get stats
                preds = model.predict(valid)
                corr = pearsonr(true_labels, preds)
                rmse = np.sqrt(mean_squared_error(true_labels, preds))
                if save:
                    print(f'iteration: {i}, pearson: {corr}, rmse: {rmse}')#,file=f, flush=True)
                    a.append([corr,rmse, true_labels, preds])
                else:
                    a.append([corr,rmse])
                    print(f'iteration: {i}, pearson: {corr}, rmse: {rmse}')#,file=f, flush=True)
            c[k].append(a)
        else:
            c['name'].append([v, for_valid, cv]) # name of the fp, valid percentage, number of cv folds 
            if save:
                nm=f'/tf/notebooks/code_for_pub/_logs_as_python_files/{fp_name}_noreplicates_{for_valid}_{time.ctime()}.pickle'
                with open(nm, 'wb') as file:
                    pickle.dump(c, file)
    return c
   