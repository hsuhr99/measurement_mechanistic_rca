import math
import os
import pickle
import argparse
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest as IForest


def LOF_norm(path, x_train, y_train):
    
    results = []

    files = os.listdir(path)
    files.sort()
    
    for f in files:
        if 'choose' in f:
            print(f'Processing {f}...')

            chose_dims = pickle.load(open(path+f, 'rb'))
            if len(chose_dims)==1:
                chose_dims = chose_dims[0]
            lof_a = LocalOutlierFactor(novelty=True)
            lof_a.fit(x_train[y_train == 0])
            values_a = - lof_a.negative_outlier_factor_.copy()[lof_a.predict()>0]
            mu_a = max(0, values_a.mean(axis=0) - 1)
            std_a = values_a.std(axis=0)

            for i in range(len(chose_dims)):
                j = (i % y_train[y_train == 1].shape[0])
                sample_to_explain = x_train[y_train == 1][j: j + 1]
                pos_g = - lof_a.score_samples(sample_to_explain)[0]
                pos_g = max(0, pos_g - 1)
                pos_g = math.erf((pos_g - mu_a) / (np.sqrt(2) * std_a))
                pos_g = max(0, pos_g)
                if len(chose_dims[i]) == 0:
                    pos_l = np.NaN
                    results.append([pos_g, pos_l])
                else:
                    lof_s = LocalOutlierFactor(novelty=True)
                    lof_s.fit(x_train[y_train==0][:, chose_dims[i]])
                    values_s = - lof_s.negative_outlier_factor_.copy()[lof_s.predict()>0]
                    #mu_s = values_s.mean(axis=0)
                    mu_s = max(0, values_s.mean(axis=0) - 1)
                    std_s = values_s.std(axis=0)
                    pos_l = - lof_s.score_samples(sample_to_explain[:, chose_dims[i]])[0]
                    pos_l = max(0, pos_l - 1)
                    pos_l = math.erf((pos_l - mu_s + 1e-6) / (np.sqrt(2) * std_s + 1e-6))
                    pos_l = max(0, pos_l)
                    results.append([pos_g, pos_l])
    results = pd.DataFrame(results, columns=['LOF_g', 'LOF_l'])
    return results


def LOF_norm_mod(path, x_train, y_train):
    
    results = []

    files = os.listdir(path)
    files.sort()
    
    for f in files:
        if 'choose' in f:
            print(f'Processing {f}...')

            new_points = pickle.load(open(os.path.join(path,f.replace('choose', 'new_points')), 'rb'))
            chose_dims = pickle.load(open(os.path.join(path, f), 'rb'))

            lof_a = LocalOutlierFactor(novelty=True)
            lof_a.fit(x_train[y_train == 0])
            values_a = - lof_a.negative_outlier_factor_.copy()[lof_a.predict()>0]
            mu_a = max(0, values_a.mean(axis=0) - 1)
            std_a = values_a.std(axis=0)
            thres_a = math.erf((- lof_a.offset_ - 1 - mu_a + 1e-6) / (np.sqrt(2) * std_a + 1e-6))

            for i in range(len(chose_dims)):
                j = (i % y_train[y_train == 1].shape[0])
                sample_to_explain = x_train[y_train == 1][j: j + 1]

                pos_g_b = - lof_a.score_samples(sample_to_explain)[0]
                pos_g_b = max(0, pos_g_b - 1)
                pos_g_b = math.erf((pos_g_b - mu_a) / (np.sqrt(2) * std_a))
                pos_g_b = max(0, pos_g_b)

                if True in np.isnan(new_points[i]):
                    pos_g_a = pos_g_b
                else:
                    pos_g_a = - lof_a.score_samples(new_points[i])[0]
                    pos_g_a = max(0, pos_g_a - 1)
                    pos_g_a = math.erf((pos_g_a - mu_a) / (np.sqrt(2) * std_a))
                    pos_g_a = max(0, pos_g_a)

                if len(chose_dims[i]) == 0:
                    #print('NaN!')
                    pos_l_b = pos_l_a = thres_l = 0.0
                    results.append([pos_g_b, pos_g_a, pos_l_b, pos_l_a, thres_a, thres_l, 0])
                else:
                    lof_s = LocalOutlierFactor(novelty=True)
                    lof_s.fit(x_train[y_train==0][:, chose_dims[i]])
                    values_s = - lof_s.negative_outlier_factor_.copy()[lof_s.predict()>0]
                    #mu_s = values_s.mean(axis=0)
                    mu_s = max(0, values_s.mean(axis=0) - 1)
                    std_s = values_s.std(axis=0)
                    thres_l = math.erf((- lof_s.offset_ - 1 - mu_s + 1e-6) / (np.sqrt(2) * std_s + 1e-6))  # thresh

                    #print(f'Mu s: {mu_s}, std s: {std_s}, np: {np.sum(lof_s.predict()>0)}, thresh_l: {thres_l}, dim: {len(chose_dims[i])}')
                    #if(std_s == 0):
                    #    print(values_s)
                    #    print(chose_dims[i])
                    #    print(i)
                    #    break
                    pos_l_b = - lof_s.score_samples(sample_to_explain[:, chose_dims[i]])[0]
                    pos_l_b = max(0, pos_l_b - 1)
                    pos_l_b = math.erf((pos_l_b - mu_s + 1e-6) / (np.sqrt(2) * std_s + 1e-6))
                    pos_l_b = max(0, pos_l_b)


                    if True in np.isnan(new_points[i]):
                        pos_l_a = pos_l_b
                        to_clean = 1
                    else:
                        pos_l_a = - lof_s.score_samples(new_points[i][:, chose_dims[i]])[0]
                        pos_l_a = max(0, pos_l_a - 1)
                        pos_l_a = math.erf((pos_l_a - mu_s + 1e-6) / (np.sqrt(2) * std_s + 1e-6))
                        pos_l_a = max(0, pos_l_a)
                        to_clean = 0
                    results.append([pos_g_b, pos_g_a, pos_l_b, pos_l_a, thres_a, thres_l, to_clean])
                    
    results = pd.DataFrame(results, columns=['LOF_g_b', 'LOF_g_a', 'LOF_l_b', 'LOF_l_a', 'thres_g', 'thres_l', 'to_clean'])
    return results


def iForest_score(path, x_train, y_train):

    if_s = IForest()
    if_s.fit(x_train, contamination=y_train.mean())
    #scores = if_s.decision_scores_
    
    results = []

    files = os.listdir(path)
    files.sort()
    
    for f in files:
        if 'choose' in f:
            print(f'Processing {f}...')
            chose_dims = pickle.load(open(os.path.join(path, f), 'rb'))
            
            if len(chose_dims) == 1:
                chose_dims = chose_dims[0]
            
            for i in range(len(chose_dims)):
                j = (i % y_train[y_train == 1].shape[0])
                sample_to_explain = x_train[y_train == 1][j: j + 1]
                #pos_g = scores[y_train == 1][j: j + 1]
                pos_g = - if_s.score_samples(sample_to_explain)[0]

                if len(chose_dims[i]) == 0:
                    pos_l = np.NaN
                    results.append([pos_g, pos_l, 0.0])
                else:
                    if_s_s = IForest(contamination=y_train.mean())
                    if_s_s.fit(x_train[:, chose_dims[i]])
                    #scores_s = if_s_s.decision_scores_
                    #pos_l = scores_s[y_train == 1][j: j + 1]
                    pos_l = - if_s_s.score_samples(sample_to_explain[:, chose_dims[i]])[0]
                    results.append([pos_g, pos_l, if_s_s.offset_])
                    
    results = pd.DataFrame(results, columns=['LOF_g', 'LOF_l', 'thresh'])
    return results, if_s.offset_


def iForest_score_our(path, x_train, y_train):
    if_s = IForest()
    if_s.fit(x_train)

    results = []

    files = os.listdir(path)
    files.sort()
    
    for f in files:
        if ('choose' in f) and ('SOD' not in f):
            print(f'Processing {f}...')
            chose_dims = pickle.load(open(os.path.join(path, f), 'rb'))
            new_points = pickle.load(open(os.path.join(path, f.replace('choose', 'new_points')), 'rb'))
            
            for i in range(len(chose_dims)):
                new_sample = new_points[i]
                sample_to_explain = x_train[y_train == 1][i: i + 1]

                pos_g_a = - if_s.score_samples(new_sample)[0]
                pos_g_b = - if_s.score_samples(sample_to_explain)[0]

                if len(chose_dims[i]) == 0:
                    pos_l_b = pos_l_a = np.NaN
                    results.append([pos_g_b, pos_g_a, pos_l_b, pos_l_a, pos_g_b - pos_g_a, pos_l_b - pos_l_a, 0.0])
                else:
                    if_s_s = IForest()
                    if_s_s.fit(x_train[:, chose_dims[i]])
                    pos_l_a = - if_s_s.score_samples(new_sample[:, chose_dims[i]])[0]
                    pos_l_b = - if_s_s.score_samples(sample_to_explain[:, chose_dims[i]])[0]
                    results.append([pos_g_b, pos_g_a, pos_l_b, pos_l_a, pos_g_b - pos_g_a, pos_l_b - pos_l_a, if_s_s.offset_])
    
    results = pd.DataFrame(results, columns=['LOF_g_b', 'LOF_g_a', 'LOF_l_b', 'LOF_l_a', 'delta_g', 'delta_l', 'thresh'])
    return results, if_s.offset_