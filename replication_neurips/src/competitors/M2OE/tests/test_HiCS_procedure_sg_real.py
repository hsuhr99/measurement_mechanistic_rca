import logging
import pickle
from itertools import chain
from time import time
import os
from sklearn import preprocessing

import numpy as np
import tensorflow as tf
import pandas as pd
from pyod.models.sod import SOD
from sklearn.neighbors import NearestNeighbors
from math import log2

from explainers.TabularMM_groups import TabularMM_groups
from utils.utils import calcola_knn_score, outliers_nighbourhood

from sklearn.ensemble import IsolationForest as IForest


def run_test(path, **kwargs):
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    loss_weights = kwargs.pop('loss_weights')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    n_runs = kwargs.pop('n_runs')
    gpu_to_use = kwargs.pop('gpu_to_use')
    dataset = kwargs.pop('dataset')

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_to_use: gpu_to_use+1], 'GPU')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    print(f'Readind file {dataset} ...')
    data = pd.read_csv(f'datasets/data/{dataset}.csv')
    x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
    y_train = data.iloc[:, -1].to_numpy()
    print('File succefully read!')
    
    in_shape = x_train.shape[1]

    # NN-model training
    near_neigh = NearestNeighbors(n_neighbors=no)
    near_neigh.fit(x_train[y_train == 0])

    base_explanations = []
    explanations = []
    explainers = []
    outliers = []
    patches_exp = []
    
    nn_points = outliers_nighbourhood(x_train, y_train, no)

    print('Base explainers')
    for ete in np.argwhere(y_train==1):
        print(f'EXPLAINING POINT: {ete}')

        x_normal = nn_points[ete[0]]
        
        mm_data_o = np.full_like(x_normal, fill_value=x_train[ete])
        mm_data_i = x_normal
        mm_data = [mm_data_o, mm_data_i]
        print(f'MM_DATA SHAPE: {mm_data_i.shape}')
        
        i = 0
        feats = []
        while i<3 and len(feats)==0:
            print('i: ', i)
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

            explainer = TabularMM_groups(in_shape, x_train[y_train == 0], loss_weights) #x_normal
            explainer.compile(optimizer=opt, run_eagerly=True)

            explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=(epochs_exp)*(x_train.shape[1]//10), verbose=0) #epochs_exp

            patches, choose = explainer.explain([mm_data[0], mm_data[1]], threshold, combine=False)
            choose = choose[0]
            print(choose)
            feats = np.argwhere(choose>0).reshape(-1)

            i += 1
            

        base_explanations.append(explainer)
        explainers.append(explainer)
        explanations.append(choose)
        patches_exp.append(patches)
        outliers.append(ete)
    selected = []
    couples = []

    for it in range(y_train[y_train==1].shape[0]-1):
        print(f'STEP :{it}')
        best_fit = -1
        best_s = -1
        best_j = -1
        for s in range(len(outliers)):
            for j in range(len(outliers)):
                if s != j and (s not in selected) and (j not in selected):
                    # ------------------- build data --------------------
                    x_normal = nn_points[outliers[j][0]]
                    mm_data_o = np.full_like(x_normal, fill_value=x_train[outliers[j][0]][0])
                    mm_data_i = x_normal

                    for op in range(1, len(outliers[j])):
                        x_normal = nn_points[outliers[j][op]]

                        mm_data_o = np.append(mm_data_o, np.full_like(x_normal, x_train[outliers[j][op]]), axis=0)
                        mm_data_i = np.append(mm_data_i, x_normal, axis=0)
                    mm_data = [mm_data_o, mm_data_i]
                    # -----------------------------------------------------
                    
                    
                    score = explainers[s].loss_fn(mm_data)
                    #print('----', s, j, score)
                    if score < best_fit or best_fit == -1:
                        best_fit = score
                        best_s = s
                        best_j = j

        #outliers.append(np.append(outliers[best_s][:], outliers[best_j][:], axis=0))
        couples.append([best_s, best_j])
        #print(f'BEST SCORE: {best_fit}')
        print(f'BEST MATCH:  {outliers[best_s]}-{explanations[best_s]}, {outliers[best_j]}-{explanations[best_j]}')
        selected.append(best_s)
        selected.append(best_j)

        # Resume explanation training
        x_normal = nn_points[outliers[best_s][0]]
        mm_data_o = np.full_like(x_normal, fill_value=x_train[outliers[best_s][0]][0])
        mm_data_i = x_normal

        for op in range(1, len(outliers[best_s])):
            x_normal = nn_points[outliers[best_s][op]]

            mm_data_o = np.append(mm_data_o, np.full_like(x_normal, x_train[outliers[best_s][op]]), axis=0)
            mm_data_i = np.append(mm_data_i, x_normal, axis=0)

        for op in range(len(outliers[best_j])):
            x_normal = nn_points[outliers[best_j][op]]

            mm_data_o = np.append(mm_data_o, np.full_like(x_normal, x_train[outliers[best_j][op]]), axis=0)
            mm_data_i = np.append(mm_data_i, x_normal, axis=0)

        mm_data = [mm_data_o, mm_data_i]
        print(f'MM_DATA SHAPE: {mm_data_i.shape}')

        # Train the explainer from scratch
        i = 0
        feats = []
        while i<3 and len(feats)==0:
            print('i: ', i)
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
            explainer = TabularMM_groups(in_shape, x_train[y_train == 0], loss_weights) #x_normal
            explainer.compile(optimizer=opt, run_eagerly=True)
            explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=(epochs_exp)*(x_train.shape[1]//10), verbose=0)#epochs_exp

            patches, choose = explainer.explain(mm_data, threshold, combine=False)
            choose = choose[0]
            print(choose)
            feats = np.argwhere(choose>0).reshape(-1)
            
            i += 1
        

        explainers.append(explainer)
        explanations.append(choose)
        patches_exp.append(patches)
        outliers.append(np.append(outliers[best_s], outliers[best_j], axis=0))
        

    pickle.dump(couples, open(os.path.join(path, 'couples.joblib'), 'wb'))
    pickle.dump(explanations, open(os.path.join(path, 'explanations.joblib'), 'wb'))
    pickle.dump(patches_exp, open(os.path.join(path, 'patches_exp.joblib'), 'wb'))
    pickle.dump(selected, open(os.path.join(path, 'selected.joblib'), 'wb'))
    pickle.dump(outliers, open(os.path.join(path, 'outliers.joblib'), 'wb'))

    #for i in range(len(explainers)):
    #    explainers[i].save_weights(os.path.join(path, f'exp_{i}.hdf5'))
    explainers[-1].save_weights(os.path.join(path, f'exp_last.hdf5'))
                
            

