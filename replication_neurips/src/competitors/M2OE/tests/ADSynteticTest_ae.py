import logging
import pickle
from math import ceil
from time import time

import sklearn
from sklearn.neighbors import NearestNeighbors

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from mpl_toolkits.mplot3d import Axes3D

from explainers import AETabularMMmd_ae
from utils.data_generator import DataGenerator
from utils.utils import getclass, getneighds, focal_loss, data_plot
from models.ad_models import define_ad_model, build_ae


def run_test(path, **kwargs):

    epochs = kwargs.pop('epochs')
    epochs_exp = kwargs.pop('exp_epochs')
    batch_exp = kwargs.pop('exp_batch')
    batch_size = kwargs.pop('batch_size')
    n_norm = kwargs.pop('n_samples_num')
    n_anorm = kwargs.pop('a_samples_num')
    n_adv = kwargs.pop('n_adv')
    n_dim = kwargs.pop('dim_number')
    # loss_weights = [1., 0.4, .5]
    loss_weights = kwargs.pop('loss_weights')
    n_mean = kwargs.pop('n_mean')
    n_std = kwargs.pop('n_std')
    dist_std = kwargs.pop('dist_std')
    ns = kwargs.pop('n_same')
    no = kwargs.pop('n_other')
    lr = kwargs.pop('learning_rate')
    anom_dims = kwargs.pop('anom_dims')
    threshold = kwargs.pop('threshold')

    logging.basicConfig(filename=os.path.join(path, 'run_log.log'), format='%(message)s', level=logging.INFO)

    # Generate a syntetic dataset with mean 3 and variance 2
    x_train = np.random.normal(n_mean, n_std, (n_norm + n_anorm, n_dim))
    dims = np.random.permutation(np.arange(n_dim))
    # Create an anomalous sample
    logging.info('\nPoint before anormality introduction: \n' + str(x_train[-1]))
    #for d in anom_dims:
    mod_dir = np.random.randint(0, 2, (anom_dims)) * 2 - 1
    logging.info(f'Mod dir dim {dims[:anom_dims]}: {mod_dir}\n')
    x_train[n_norm:, dims[:anom_dims]] += mod_dir * dist_std * n_std
    x_train = x_train.astype(np.float32)

    print('DIMS: ', dims[:anom_dims])

    logging.info('\nPoint after anormality introduction: \n' + str(x_train[-1]))

    y_train = np.zeros(x_train.shape[0], dtype=np.int32)
    y_train[n_norm:] = 1
    print(x_train[y_train==1].shape)
    y_train = y_train.astype(np.float32)

    print('MIN BS', x_train.min())
    print('MAX BS', x_train.max())

    #scaler = sklearn.preprocessing.Normalizer()
    #scaler = scaler.fit(x_train[y_train==0])
    #x_train = scaler.transform(x_train)


    print('MIN AS', x_train[y_train==0].min(axis=0))
    print('MAX AS', x_train[y_train==0].max(axis=0))
    print('MIN AS ANOM', x_train[y_train==1].min(axis=0))
    print('MAX AS ANOM', x_train[y_train==1].max(axis=0))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x_train[y_train==0, 0], x_train[y_train==0, 1], x_train[y_train==0, 2], label='normal')
    # ax.scatter3D(x_train[y_train==1, 0], x_train[y_train==1, 1], x_train[y_train==1, 2], label='anomalous', c='red')
    # ax.legend()
    # plt.show()

    data_plot(x_train, y_train)

    in_shape = x_train.shape[1]

    # Define the ad model
    ad_model = build_ae(in_shape)

    lr_schedule_ad = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=epochs*n_adv, decay_rate=0.3,
                                                                    staircase=True)
    lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=epochs_exp, decay_rate=0.7,
                                                                     staircase=True)
    ad_model_opt = tf.keras.optimizers.Adam()
    exp_opt_c = tf.keras.optimizers.Adam(learning_rate=lr)
    exp_opt_m = tf.keras.optimizers.Adam(learning_rate=lr)


    # first train
    #ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)
    ad_model.compile(optimizer=ad_model_opt, loss=focal_loss)
    ad_model.fit(x_train[y_train == 0], x_train[y_train == 0], batch_size=batch_size, epochs=epochs, verbose=1, )
    ad_model.trainable = False

    # print(ad_model.evaluate(x_train, y_train))
    # print(np.where(ad_model.predict(x_train)[:, 0] < 0.5, 0, 1))

    results = []
    explanations = []
    masks = []

    # -------------------- Try with more samples ---------------------------
    near_neigh = NearestNeighbors(n_neighbors=no)
    near_neigh.fit(x_train[y_train == 0])
    _, y_normal = near_neigh.kneighbors(x_train[y_train==0])
    x_normal = x_train[y_train == 0][y_normal[0]]

    mm_data_o = np.full_like(x_normal, fill_value=x_train[y_train==1])
    mm_data_i = x_normal
    mm_data = [mm_data_o, mm_data_i]


    explainer = AETabularMMmd_ae.TabularMM(ad_model, in_shape, x_train[y_train == 0], x_train[y_train==1],
                                           optimizer_m=exp_opt_m, optimizer_c=exp_opt_c, weights=None, show=True)

    print(explainer.MASKGEN.summary())
    print(explainer.PATCH.summary())
    # ----------------------------------------------------------------------

    for i in range(n_adv):
        print('--------------------- ADV EPOCH: {} -------------------'.format(i))
        sample_to_explain = x_train[np.where(y_train == 1)]

        start_time = time()
        # Early-stopping

        exp_opt = tf.keras.optimizers.Adam(learning_rate=lr)
        explainer.explain(mm_data, None, batch_size=batch_exp,
                          epochs=epochs_exp, loss_weights=loss_weights,
                          optimizer=exp_opt)  # loss_weights=[1., 0.2, .4]
        tot_time = time() - start_time
        print('Elapsed time: ', tot_time)
        logging.info(f'Elapsed time explanation {i}: {tot_time}')

        new_sample, choose = explainer.return_explanation([mm_data[0], mm_data[1]], threshold)
        choose = np.argwhere(choose >= threshold).reshape(-1)

        d_found = set(choose)
        d_real = set(dims[:anom_dims])
        sens = len(d_real.intersection(d_found)) / len(d_real)
        if len(d_found) > 0:
            prec = len(d_real.intersection(d_found)) / len(d_found)
        else:
            prec = 0

        #data_plot(x_train, y_train, new_point=new_sample[0], name=os.path.join(path, f'adv_point_{i}'), train=True)
        #data_plot(mm_data, test_labels_expl, new_point=new_sample[0], name=os.path.join(path, f'neighbourhood_{i}'), train=True)

        #plt.figure(2, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        #plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        #plt.bar(np.arange(0, n_dim, dtype=np.int32), mask[0])
        #plt.title('Try n.: {}'.format(i + 1))
        #plt.tight_layout()
        #plt.figure(3, figsize=(4 * min(n_adv, 5), 3.8 * ceil(n_adv / 5)))
        #plt.subplot(ceil(n_adv / 5), min(n_adv, 5), i + 1)
        #plt.bar(np.arange(0, n_dim, dtype=np.int32), choose[0])
        #plt.title('Try n.: {}'.format(i + 1))
        #plt.tight_layout()

    print('SENS: ', sens)
    print('PREC: ', prec)

    pickle.dump(mm_data, open(os.path.join(path, 'chosen_points_x.joblib'), 'wb'))
    pickle.dump(results, open(os.path.join(path, 'results.joblib'), 'wb'))
    pickle.dump(masks, open(os.path.join(path, 'masks.joblib'), 'wb'))
    #plt.figure(2)
    #plt.savefig(os.path.join(path, 'explanations.eps'))
    #plt.savefig(os.path.join(path, 'explanations.jpg'))
    #plt.figure(3)
    #plt.savefig(os.path.join(path, 'choose.eps'))
    #plt.savefig(os.path.join(path, 'choose.jpg'))
    #plt.show()
    new_sample, dims = explainer.return_explanation([mm_data[0], mm_data[1]], threshold=threshold)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x_train[y_train == 0, 0], x_train[y_train == 0, 1], x_train[y_train == 0, 2], label='normal')
    # ax.quiver(x_train[y_train == 1, 0], x_train[y_train == 1, 1], x_train[y_train == 1, 2],
    #           -x_train[y_train == 1, 0] + new_sample[:, 0], -x_train[y_train == 1, 1] + new_sample[:, 1],
    #           -x_train[y_train == 1, 2] + new_sample[:, 2],
    #           arrow_length_ratio=0.1
    #           )
    # ax.scatter3D(x_train[y_train == 1, 0], x_train[y_train == 1, 1], x_train[y_train == 1, 2], label='anomalous', c='red')
    # ax.scatter3D(new_sample[:,0], new_sample[:,1], new_sample[:,2], label='patched sample', c='orange')
    # ax.legend()
    # plt.show()

    #data_plot(x_train, y_train, new_point=new_sample[0], dimensions=np.where(dims > threshold)[1],
    #          name=os.path.join(path, f'explanation'), train=True)
    pickle.dump(np.argwhere(dims>=threshold), open('choose.pickle', 'wb'))
    plt.close('all')