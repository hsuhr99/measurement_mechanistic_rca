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

#from explainers import AETabularMMmd_ae
# 23/03/2023
from explainers import TabularMM_groups as AETabularMMmd_ae
from utils.utils import calcola_knn_score

from sklearn.ensemble import IsolationForest as IForest


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


def run_test(path, **kwargs):
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    loss_weights = kwargs.pop('loss_weights')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    n_runs = kwargs.pop('n_runs')
    gpu_to_use = kwargs.pop('gpu_to_use')
    data_path = 'datasets/synthetic/processed/'

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_to_use: gpu_to_use+1], 'GPU')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    files = os.listdir(data_path)
    files.sort()
    f = 'synth_multidim_075_000.csv'

    print(f'FILE: {f}')

    dims = [10, 20, 30, 40, 50, 75]
    subspaces = pickle.load(open(data_path + f.replace('synth_', 'info_synth_').replace('.csv', '.joblib'), 'rb'))
    
    data = pd.read_csv(data_path + f)
    x_train_all = data.iloc[:, :data.shape[1] - 1].to_numpy()
    y_train = data['class'].to_numpy().astype(int)

    for ac in np.unique(data['class'].to_numpy())[np.unique(data['class'].to_numpy())>0]:
        subpath_ac = os.path.join(path, str(int(ac)))
        os.mkdir(subpath_ac)
        
        sub = np.array(subspaces[int(log2(ac))], dtype=int)
        print(f'SUB: {sub}')
        s_feats = np.random.permutation(list(set(range(x_train_all.shape[1])).difference(set(sub))))
        pickle.dump(s_feats, open(os.path.join(subpath_ac, f'feats_{int(ac)}.joblib'), 'wb'))
        
        for r in dims:
            print(f'DIMS {r}')
            subpath = os.path.join(subpath_ac, str(r))
            os.mkdir(subpath)
            

            ds_feats = np.append(sub, s_feats[:r-len(sub)])
            print(ds_feats)
            in_shape = len(ds_feats)

            x_train = x_train_all[:, ds_feats]

            # Create the data set with less dimensions

            near_neigh = NearestNeighbors(n_neighbors=no)
            near_neigh.fit(x_train[y_train == 0])

            base_explanations = []
            explanations = []
            explainers = []
            outliers = []
            
            nn_points = outliers_nighbourhood(x_train, y_train, no)

            print('Base explainers')
            for ete in np.argwhere(data['class'].to_numpy() == ac):
                print(f'EXPLAINING POINT: {ete}')

                #x_train_sub = x_train[y_train == 0]
                #x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
                #y_train_sub = y_train[y_train == 0]
                #y_train_sub = np.append(y_train_sub, [1.], axis=0)

                # -------------------- Try with more samples ---------------------------
                x_normal = nn_points[ete[0]]
                print('SHAPE: ', x_normal.shape)

                mm_data_o = np.full_like(x_normal, fill_value=x_train[ete])
                mm_data_i = x_normal
                mm_data = [mm_data_o, mm_data_i]

                opt = tf.keras.optimizers.Adam(learning_rate=lr)

                explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train == 0], loss_weights) #x_train[y_train == 0]
                explainer.compile(optimizer=opt, run_eagerly=True)

                explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=(r//10)*epochs_exp, verbose=0) #epochs_exp

                _, choose = explainer.explain([mm_data[0], mm_data[1]], threshold, combine=False)
                choose = choose[0]
                feats = np.argwhere(choose>0).reshape(-1)

                base_explanations.append(explainer)
                explainers.append(explainer)
                explanations.append(choose)
                outliers.append(ete)
                selected = []
                couples = []

            for it in range(data[data['class'].to_numpy() == ac].shape[0]-1):
                print(f'STEP :{it}')
                best_fit = -1
                best_s = -1
                best_j = -1
                for s in range(len(outliers)):
                    for j in range(len(outliers)):
                        if s != j and (s not in selected) and (j not in selected):
                            x_normal = nn_points[outliers[j][0]]
                            mm_data_o = np.full_like(x_normal, fill_value=x_train[outliers[j][0]][0])
                            mm_data_i = x_normal

                            for op in range(1, len(outliers[j])):
                                x_normal = nn_points[outliers[j][op]]

                                mm_data_o = np.append(mm_data_o, np.full_like(x_normal, x_train[outliers[j][op]]), axis=0)
                                mm_data_i = np.append(mm_data_i, x_normal, axis=0)
                            mm_data = [mm_data_o, mm_data_i]
                            
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
                #print(f'MM_DATA SHAPE: {mm_data_o.shape}')

                # Train explainer from scratch
                explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train == 0], loss_weights)
                explainer.compile(optimizer=opt, run_eagerly=True)
                explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=(r//10)*epochs_exp, verbose=0)
                _, choose = explainer.explain([mm_data[0], mm_data[1]], threshold, combine=False)
                choose = choose[0]
                feats = np.argwhere(choose>0).reshape(-1)

                explainers.append(explainer)
                explanations.append(choose)
                outliers.append(np.append(outliers[best_s], outliers[best_j], axis=0))
                

            
            print(f'PREC: {len(set(feats).intersection(set(range(len(sub)))))/len(feats)}')
            print(f'REC: {len(set(feats).intersection(set(range(len(sub)))))/len(sub)}')

            pickle.dump(couples, open(os.path.join(subpath, 'couples.joblib'), 'wb'))
            pickle.dump(explanations, open(os.path.join(subpath, 'explanations.joblib'), 'wb'))
            pickle.dump(selected, open(os.path.join(subpath, 'selected.joblib'), 'wb'))
            pickle.dump(outliers, open(os.path.join(subpath, 'outliers.joblib'), 'wb'))

            for i in range(len(explainers)):
                explainers[i].save_weights(os.path.join(subpath, f'exp_{i}.hdf5'))
                
            

