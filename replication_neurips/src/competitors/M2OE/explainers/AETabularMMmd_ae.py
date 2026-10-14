from math import ceil

from tensorflow.keras.layers import ReLU, BatchNormalization


from ..explainers.MaskingModelExplainer import MaskingModelExplainer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Activation, Concatenate
import numpy as np


class TabularMM(MaskingModelExplainer):

    def __init__(self, in_shape, normal_data, loss_weights):
        # print("Reached TabMM")
        super(TabularMM, self).__init__(loss_weights)
        self.buildExplanator(in_shape)
        # print("Reached TabMM 2")
        # print(normal_data.shape)
        differences = (- normal_data[:, np.newaxis, :] + normal_data[np.newaxis, :, :])**2
        self.normal_dist = differences.sum(axis=1) / (differences.shape[1]-1)
        self.normal_dist = self.normal_dist.mean(axis=0)
        # print("Reached TabMM 3")

        #print('NORMAL DIST SHAPE: ', self.normal_dist.shape)

    def call(self, inputs, training=None, mask=None):
        #masks, choose = self.MASKGEN(inputs)
        masks = self.MASK(inputs)
        choose = self.CHOOSE(inputs)
        patches = self.MASKAPPLY([inputs[0], masks, choose])
        return patches


    def defineMaskGen(self, in_shape):

        #num_unit = 64 * (in_shape//30 + 1)
        #num_unit = 32 * (in_shape//10) # 2**((in_shape//10)-1)
        # 23/03/2023
        in_shape_int = in_shape[0]
        num_unit = in_shape_int * 4 #3
        # print(num_unit)

        input_o = Input(in_shape)
        input_i = Input(in_shape)
        inputs = Concatenate()([input_o, input_i])

        x1 = Dense(num_unit, activation='relu')(inputs)
        x1 = Dense(num_unit, activation='relu')(x1)
        outputs_c = Dense(in_shape_int, activation='sigmoid')(x1)
        self.CHOOSE = keras.Model(inputs=[input_o, input_i], outputs=outputs_c, name='CHOOSE')

        x0 = Dense(num_unit)(inputs)
        x0 = Dense(num_unit)(x0)
        outputs = Dense(in_shape_int)(x0)
        self.MASK = keras.Model(inputs=[input_o, input_i], outputs=outputs, name='MASK')

        return

    def defineMaskApply(self, in_shape):
        inputs = [Input(in_shape, name='input_img'), Input(in_shape, name='input_mask'),
                  Input(in_shape, name='input_choice')]  # Sample, Mask
        mid_output = Multiply()([inputs[1], inputs[2]]) # TODO prima era **2

        outputs = Add()([inputs[0], mid_output])
        self.MASKAPPLY = keras.Model(inputs=inputs, outputs=outputs)

        return

    def train_step(self, data):
        x, y = data         # y --> normal data
        data_o = x[0]
        data_i = x[1]

        with tf.GradientTape() as tape:
            #mask = self.MASK([data_o, data_i])
            #choose = self.CHOOSE([data_o, data_i])
            #patches = self.MASKAPPLY([data_o, mask, choose])
            # 23/03/2023 H 16:38 test per vedere se megari come scritto sopra avevo creato qualche problema nell'aggiornamento
            patches, mask, choose = self.PATCH([data_o, data_i])
            #print(choose)

            ndim_loss = tf.sqrt(tf.reduce_sum(choose ** 2, axis=1))
            
            # 27/05/2023 aggiunto (* choose ) per vedere se così le machere diventano dipendenti dalla scelta
            margin_n = tf.sqrt(tf.reduce_sum(((patches - data_i) ** 2) * choose, axis=1)) / np.sqrt(data_o.shape[1]) # y aggiunta oggi

            differences = (- data_o + data_i)
            #print(differences)
            differences_red = tf.reduce_sum((differences ** 2) * (choose ** 2), axis=1)
            ## 25/04/2023 inserita radice quadrata
            #sample_distance = tf.sqrt(tf.reduce_sum(self.normal_dist * (choose * 2), axis=1) / (differences_red + 1e-4))
            sample_distance = tf.sqrt(tf.reduce_sum(self.normal_dist * (choose), axis=1) / (differences_red + 1e-4))
            
            # 08/05/2023 Modifica loss
            #differences = self.normal_dist / (differences + 1e-6)
            #print(differences)
            #print(differences)
            #print(choose)
            #sample_distance = differences * (choose ** 2)
            #sample_distance = tf.sqrt(tf.reduce_sum(sample_distance, axis=1))
            
            #print(tf.reduce_mean(margin_n), tf.reduce_mean(sample_distance), tf.reduce_mean(ndim_loss))
            

            loss = tf.reduce_mean(self.loss_weights[0] * margin_n +
                                  self.loss_weights[1] * sample_distance +
                                  self.loss_weights[2] * ndim_loss)

        gradients = tape.gradient(loss, self.PATCH.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.PATCH.trainable_variables))

        self.compiled_metrics.update_state(y, patches)
        return {m.name: m.result() for m in self.metrics}

    # 24/03/2023 per testare cosa succede se abbassiamo la threshold
    def explain(self, sample, threshold=0.7, acceptance_ratio=0.5, combine=True):

        # compute model output
        mask = self.MASK(sample)
        choose = np.mean(self.CHOOSE(sample),axis=0)
        # if not tf.is_tensor(mask):
        #     choose = tf.convert_to_tensor(mask, dtype=tf.float32)
        # if not tf.is_tensor(choose):
        #     choose = tf.convert_to_tensor(choose, dtype=tf.float32)

        return choose
        if combine:
            mask = tf.reduce_mean(mask, axis=0)
            choose = tf.where(choose.numpy() > threshold, 1, 0)

            print('RATIO: ', np.mean(choose, axis=0))
            choose = tf.where(np.mean(choose, axis=0)>acceptance_ratio, 1, 0)
            mask = tf.expand_dims(mask, axis=0)  # add batch dim
            choose = tf.expand_dims(choose, axis=0)

            patches = self.MASKAPPLY([sample[0][0:1], mask, choose])
        else:
            mask = tf.expand_dims(mask, axis=0)  # add batch dim

            choose = tf.where(choose.numpy() > threshold, 1, 0)
            choose = tf.expand_dims(choose, axis=0)

            patches = self.MASKAPPLY([sample[0], mask, choose])

        return patches.numpy(), choose
