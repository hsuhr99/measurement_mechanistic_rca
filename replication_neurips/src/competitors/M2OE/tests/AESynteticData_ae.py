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

#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.set_visible_devices(physical_devices[0:1], 'GPU')


def run_test(path, **kwargs):
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    loss_weights = kwargs.pop('loss_weights')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    threshold = kwargs.pop('threshold')
    data_path = 'datasets/synthetic/processed/'

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), level=logging.INFO, format='%(message)s')

    precisions = []
    precisions_sod = []
    sensibilities = []
    distances = []
    distances_sod = []

    files = os.listdir(data_path)
    files.sort()
    print(files)
    for f in files: #[45:]
        if not f.startswith('info_'):
            logging.info(f'FILE: {f}')
            print(f'FILE: {f}')

            data = pd.read_csv(data_path + f)
            x_train = data.iloc[:, :data.shape[1] - 1].to_numpy()
            y_train = data['class'].to_numpy()
            
            in_shape = x_train.shape[1]

            near_neigh = NearestNeighbors(n_neighbors=no)
            near_neigh.fit(x_train[y_train==0])

            f_choose = []
            sod_choose = []

            positions = []
            positions_sod = []
            new_points = []

            for ete in np.argwhere(data['class'].to_numpy() != 0):

                x_train_sub = x_train[y_train == 0]
                x_train_sub = np.append(x_train_sub, x_train[ete], axis=0)
                y_train_sub = y_train[y_train == 0]
                y_train_sub = np.append(y_train_sub, [1.], axis=0)

                # -------------------- Try with more samples ---------------------------
                _, y_normal = near_neigh.kneighbors(x_train[ete])
                x_normal = x_train[y_train==0][y_normal[0]]
                print('SHAPE: ', x_normal.shape)


                mm_data_o = np.full_like(x_normal, fill_value=x_train[ete])
                mm_data_i = x_normal
                mm_data = [mm_data_o, mm_data_i]

                opt = tf.keras.optimizers.Adam(learning_rate=lr)

                explainer = AETabularMMmd_ae.TabularMM(in_shape, x_train[y_train==0], loss_weights)
                explainer.compile(optimizer=opt, run_eagerly=True)

                logging.info(f'EXPLAINING POINT: {ete}')
                print(f'EXPLAINING POINT: {ete}')

                start_time = time()
                explainer.fit(mm_data, mm_data_i, batch_size=batch_exp, epochs=epochs_exp, verbose=0)
                tot_time = time() - start_time

                print('Elapsed time: ', tot_time)
                logging.info(f'Elapsed time explanation: {tot_time}')



                # Use SOD
                sod = SOD(n_neighbors=40, ref_set=20, alpha=0.9)
                sod.fit(x_train_sub)
                choose_s = _sod_2(sod, x_train_sub, x_train_sub.shape[0] - 1)
                sod_choose.append(choose_s)

                # Our method
                new_sample, choose = explainer.explain([mm_data[0], mm_data[1]], threshold)
                choose = np.argwhere(choose >= threshold).reshape(-1)
                print('CHOSEN: ', choose)
                #print('NEW POINT: ', new_sample)
                #print('ORIGINAL POINT: ', mm_data[0][0])
                f_choose.append(choose)
                new_points.append(new_sample)

                # Compare results
                if len(choose)>0:
                    dists, pos, _ = calcola_knn_score(x_train_sub[:, choose], y_train_sub, x_train_sub.shape[0]-1)
                else:
                    dists = -1
                    pos = -1
                print(choose_s)
                dists_sod, pos_sod, _ = calcola_knn_score(x_train_sub[:, choose_s], y_train_sub, x_train_sub.shape[0] - 1)
                positions.append(pos)
                positions_sod.append(pos_sod)
                print(pos)
                print(pos_sod)
                distances.append(dists)
                distances_sod.append(dists_sod)

            pickle.dump(f_choose, open(os.path.join(path, f.replace('info_', '').replace('.csv', '_')+'choose.joblib'), 'wb'))
            pickle.dump(sod_choose, open(os.path.join(path, f.replace('info_', '').replace('.csv', '_')+'sod_choose.joblib'), 'wb'))
            pickle.dump(new_points,
                        open(os.path.join(path, f.replace('info_', '').replace('.csv', '_') + 'new_points.joblib'),
                             'wb'))

            info_df = pd.read_csv(data_path+f.replace('synth_', 'info_synth_'))
            subspaces = pickle.load(open(data_path+f.replace('synth_', 'info_synth_').replace('.csv', '.joblib'), 'rb'))
            prec_f = []
            sens_f = []

            for i in range(len(f_choose)):
                features = np.array(subspaces)[list(info_df.iloc[i, :info_df.shape[1]-1])]
                d_found = set(f_choose[i])
                d_real = set(list(chain.from_iterable(features)))
                sens = len(d_real.intersection(d_found)) / len(d_real)
                if len(d_found) > 0:
                    prec = len(d_real.intersection(d_found)) / len(d_found)
                else:
                    prec = 0
                prec_f.append(prec)
                sens_f.append(sens)
            precisions.append(prec_f)
            sensibilities.append(sens_f)
            pickle.dump(sens_f, open(os.path.join(path, f.replace('.csv', '_test_sensibilities.joblib')), 'wb'))
            pickle.dump(prec_f, open(os.path.join(path, f.replace('.csv', '_test_precisions.joblib')), 'wb'))
            print(f'Precision: {np.array(prec_f).mean()}')
            print(f'Sensitibility: {np.array(sens_f).mean()}')

            print(
                f"Ours: {np.where(np.array(positions) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))} ",
                f"SOD: {np.where(np.array(positions_sod) == 0, 1, 0).sum() / len(np.argwhere(data['class'].to_numpy() != 0))}")
        else:
            print(f'File {f} skipped!')


    pickle.dump(precisions_sod, open(os.path.join(path, 'test_precisions_sod.joblib'), 'wb'))
    pickle.dump(precisions, open(os.path.join(path, 'test_precisions.joblib'), 'wb'))
    pickle.dump(distances_sod, open(os.path.join(path, 'test_distances_sod.joblib'), 'wb'))
    pickle.dump(distances, open(os.path.join(path, 'test_distances.joblib'), 'wb'))
    print(f'Mean precision: {np.array(precisions).mean()}')
    print(f'Mean sensitibility: {np.array(precisions_sod).mean()}')