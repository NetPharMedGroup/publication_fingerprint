from catboost import Pool, CatBoostRegressor
from collections import defaultdict
import copy
from scipy.stats import pearsonr
import datetime
from mymodule import make_combo_fp, predict_synergy
import numpy as np
import pandas as pd
import pickle
from progiter import ProgIter
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn.svm import SVR, LinearSVR
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate, cross_val_score
from xgboost import XGBRegressor
import time
import warnings

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

def scoring_func_p(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]

def scoring_func_m(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

pearson = make_scorer(score_func=scoring_func_p, 
                      greater_is_better=True)
rmse = make_scorer(score_func=scoring_func_m, 
                   greater_is_better=False)

def run_train_all_sklearn(file, fp_name, cv=5, verbose=0, seed=1):
    
    np.random.seed(seed)
    c = defaultdict(list)
    
    for k in ProgIter(['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa','css_ri','name'], verbose=verbose, total=5):
        v = file[k]
        
        if k != 'name':
            temp = dict() # for results storage. Assuming that "name" comes last
            
            if 'drug_row_col' in v.columns:
                v.drop(columns=['drug_row_col'], inplace=True)
            
            cat_cols = ['cell_line_name']
            categories = [v[column].unique() for column in v[cat_cols]] # manually find all available categories for one-hot
            
            # pipelines
            encode = Pipeline(steps=[
                ('one-hot-encode', OneHotEncoder(categories=categories))
            ])
            processor = ColumnTransformer(
                transformers=[
                    ('cat_encoding', encode, cat_cols),
                    ('dropping', 'drop', [k]) 
                ],
                remainder='passthrough')
            
            catbst = ColumnTransformer(
                transformers=[
                    ('dropping', 'drop', [k]) 
                ],
                remainder='passthrough')
            
            # regressions
            lr = make_pipeline(processor, linear_model.LinearRegression())
            ridge = make_pipeline(processor, linear_model.Ridge())
            lasso = make_pipeline(processor, linear_model.Lasso())
            elastic = make_pipeline(processor, linear_model.ElasticNet())
            lassolars = make_pipeline(processor, linear_model.LassoLars())
            b_ridge = make_pipeline(processor, linear_model.BayesianRidge())
            kernel = DotProduct() + WhiteKernel()
            gpr = make_pipeline(processor, GaussianProcessRegressor(kernel=kernel))
            linSVR = make_pipeline(processor, LinearSVR())
            hist_gbr = make_pipeline(processor, HistGradientBoostingRegressor(warm_start=True, 
                                                                              max_depth=6))
            rfr = make_pipeline(processor, RandomForestRegressor(warm_start=True, 
                                                                 max_depth=6, 
                                                                 n_jobs=3))
            iso = make_pipeline(processor, IsotonicRegression(increasing='auto'))
            xgb = make_pipeline(processor, XGBRegressor(tree_method='gpu_hist', 
                                                        max_depth=6))
            cbt = make_pipeline(catbst, CatBoostRegressor(task_type='GPU', 
                                                          depth=6, 
                                                          cat_features=np.array([0]), 
                                                          verbose=False))
            
            mls = [cbt, rfr, gpr, hist_gbr,
                   lr, ridge, lasso, elastic, lassolars, 
                   b_ridge, 
                   gpr, 
                   linSVR, 
                   iso]
            mls_names = ["cbt","rfr","gpr", "hist_gbr",
                         "lr", "ridge","lasso","elastic","lassolars", 
                         "b_ridge", 
                         "gpr", 
                         "linSVR", 
                         "iso"]
            
            # results 
            start = time.time()
            for MODEL, name in zip(mls, mls_names):
                print(f'\n{name}')
                if 'cbt' == name:
                    n_jobs = 1
                else:
                    n_jobs = cv
                cv_dict = cross_validate(
                    MODEL,
                    v, 
                    v[k],
                    cv=cv, 
                    scoring={"pearsonr":pearson,
                             "rmse":rmse}, 
                    return_train_score=False, 
                    verbose=verbose,
                    n_jobs=n_jobs, 
                )
                temp[name] = {
                    'test_pearsonr' : np.nanmean(cv_dict['test_pearsonr']), 
                    'test_rmse' : abs(np.nanmean(cv_dict['test_rmse'])) 
                }
                print(temp[name])
            print(f'{k} took {int(time.time()-start)/60} mins')
            
            c[k] = temp
        else:
            nm=f'/tf/notebooks/code_for_pub/_logs_as_python_files/{fp_name}_13models_5foldCV_{time.ctime()}.pickle'
            with open(nm, 'wb') as file:
                pickle.dump(c, file)
            print(f'saving complete to {nm}')
    return c