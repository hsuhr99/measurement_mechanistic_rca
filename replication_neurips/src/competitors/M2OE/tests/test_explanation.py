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

from explainers import AETabularMMmd_ae
from utils.utils import _sod_2, calcola_knn_score


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
    #print(files)
    for f in files:  # [45:]
        if not (f.startswith('info_') or ('100' in f)):
            logging.info(f'FILE: {f}')
            print(f'FILE: {f}')
    
            data = pd.read_csv(data_path + f)
            x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
            y_train = data['class'].to_numpy()
    
            in_shape = x_train.shape[1]
    
            near_neigh = NearestNeighbors(n_neighbors=no)
            near_neigh.fit(x_train[y_train == 0])
    
            f_choose = []
    
            for ete in np.argwhere(data['class'].to_numpy() != 0):
                print(f'EXPLAINING POINT: {ete}')
    
                #x_train_sub = x_train[y_train == 0]
                #x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
                #y_train_sub = y_train[y_train == 0]
                #y_train_sub = np.append(y_train_sub, [1.], axis=0)
    
                # -------------------- Try with more samples ---------------------------
                _, y_normal = near_neigh.kneighbors(x_train[ete])
                x_normal = x_train[y_train == 0][y_normal[0]]
                print('SHAPE: ', x_normal.shape)
    
                mm_data_o = np.full_like(x_normal, fill_value=x_train[ete])
                mm_data_i = x_normal
                mm_data = [mm_data_o, mm_data_i]
    
                opt = tf.keras.optimizers.Adam(learning_rate=lr)
    
                explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train == 0], loss_weights) #x_train[y_train == 0]
                explainer.compile(optimizer=opt, run_eagerly=True)
    
                start_time = time()
                explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=epochs_exp, verbose=0) #epochs_exp
                tot_time = time() - start_time
    
                print('Elapsed time: ', tot_time)
                logging.info(f'Elapsed time explanation: {tot_time}')
    
    
                # Our method
                new_sample, choose = explainer.explain([mm_data[0], mm_data[1]], threshold, combine=False)
                #choose = np.argwhere(choose >= threshold).reshape(-1)
                #print(choose)
                f_choose.append(choose)
                explainer.save_weights(os.path.join(path, f'exp_{f[15:22]}_{ete[0]}.hdf5'))
    
            pickle.dump(f_choose, open(os.path.join(path, f.replace('info_', '').replace('.csv', '_choose.joblib')), 'wb'))

        else:
            print(f'File {f} skipped!')