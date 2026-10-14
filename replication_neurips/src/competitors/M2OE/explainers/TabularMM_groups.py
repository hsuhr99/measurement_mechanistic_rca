from math import ceil

from tensorflow.python.keras.layers import ReLU, BatchNormalization

from explainers.MaskingModelExplainer import MaskingModelExplainer
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Add, Multiply, Activation, Concatenate
import numpy as np


class TabularMM_groups(MaskingModelExplainer):

    def __init__(self, in_shape, normal_data, loss_weights):
        super(TabularMM_groups, self).__init__(loss_weights)
        self.buildExplanator(in_shape)

        differences = (- normal_data[:, np.newaxis, :] + normal_data[np.newaxis, :, :])**2
        self.normal_dist = differences.sum(axis=1) / (differences.shape[1]-1)
        self.normal_dist = self.normal_dist.mean(axis=0)

        #print('NORMAL DIST SHAPE: ', self.normal_dist.shape)

    def call(self, inputs, training=None, mask=None):
        #masks, choose = self.MASKGEN(inputs)
        ones_input = [np.ones_like(inputs[0]),np.ones_like(inputs[0])] 
        masks = self.MASK(inputs)
        choose = self.CHOOSE(ones_input)
        patches = self.MASKAPPLY([inputs[0], masks, choose])
        return patches


    def defineMaskGen(self, in_shape):

        #num_unit = 64 * (in_shape//30 + 1)
        #num_unit = 32 * (in_shape//10) # 2**((in_shape//10)-1)
        # 23/03/2023
        num_unit = in_shape * 3 #4
        print(num_unit)

        input_o = Input(in_shape)
        input_i = Input(in_shape)
        inputs = Concatenate()([input_o, input_i])

        x1 = Dense(num_unit, activation='relu')(inputs)
        x1 = Dense(num_unit, activation='relu')(x1)
        outputs_c = Dense(in_shape, activation='sigmoid')(x1)
        self.CHOOSE = keras.Model(inputs=[input_o, input_i], outputs=outputs_c, name='CHOOSE')

        x0 = Dense(num_unit)(inputs)
        x0 = Dense(num_unit)(x0)
        outputs = Dense(in_shape)(x0)
        self.MASK = keras.Model(inputs=[input_o, input_i], outputs=outputs, name='MASK')

        return

    def defineMaskApply(self, in_shape):
        inputs = [Input(in_shape, name='input_img'), Input(in_shape, name='input_mask'),
                  Input(in_shape, name='input_choice')]  # Sample, Mask
        mid_output = Multiply()([inputs[1], inputs[2]]) # TODO prima era **2

        outputs = Add()([inputs[0], mid_output])
        self.MASKAPPLY = keras.Model(inputs=inputs, outputs=outputs)

        return
    
    def definePatch(self, in_shape):
        """
        Define the model that produce the patch from the original image
        :param in_shape: input shape
        :return:
        """

        # Mette insieme maskapply e maskgen
        img_o = keras.Input(shape=in_shape)
        img_i = keras.Input(shape=in_shape)
        
        # --------------------------------
        #num_unit = in_shape * 4
        #x1 = Dense(num_unit, activation='relu')(img_o)
        #x1 = Dense(1, activation=lambda x: 1/(1+tf.math.exp(-3*x)))(x1)
        #self.WEIGHTS = keras.Model(inputs=[img_o], outputs=[x1])
        # --------------------------------

        self.defineMaskGen(in_shape)
        self.defineMaskApply(in_shape)

        mask = self.MASK([img_o, img_i])
        choose = self.CHOOSE([tf.ones_like(img_o), tf.ones_like(img_i)])
        patch = self.MASKAPPLY([img_o, mask, choose])

        # Generatore patch - Applicatore della patch - Modello completo (gen+app)
        self.PATCH = keras.Model(inputs=[img_o, img_i], outputs=[patch, mask, choose])
        return

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
             
            #w = self.WEIGHTS(data_o)
            patches, mask, choose = self.PATCH(x)
            loss = self.compute_loss(x, patches, mask, choose)
            #print(loss)
            
        gradients = tape.gradient(loss, self.PATCH.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.PATCH.trainable_variables))

        self.compiled_metrics.update_state(y, patches)
        return {m.name: m.result() for m in self.metrics}
    
    def loss_fn(self, data):
        patches, mask, choose = self.PATCH(data)
        return self.compute_loss(data, patches, mask, choose).numpy()
    
    def compute_loss(self, data, patches, mask, choose):
        data_o = data[0]
        data_i = data[1]
        
        ndim_loss = tf.sqrt(tf.reduce_sum(choose ** 2, axis=1))
        margin_n = tf.sqrt(tf.reduce_sum(((patches - data_i) ** 2) * choose, axis=1)) / np.sqrt(data_o.shape[1]) 
        differences = (- data_o + data_i)
        differences_red = tf.reduce_sum((differences ** 2) * (choose ** 2), axis=1)
        normal_dist = tf.sqrt(tf.reduce_sum(self.normal_dist * (choose), axis=1))
        sample_distance = normal_dist / (differences_red + 1e-4)
        
        loss = tf.reduce_mean(self.loss_weights[0] * margin_n +
                              self.loss_weights[1] * sample_distance +
                              self.loss_weights[2] * ndim_loss)
        
        return loss
        

    # 24/03/2023 per testare cosa succede se abbassiamo la threshold
    def explain(self, sample, threshold=0.7, acceptance_ratio=0.5, combine=True):

        # compute model output
        ones_vect = [np.ones_like(sample[0]),np.ones_like(sample[1])]
        mask = self.MASK(sample)
        choose = self.CHOOSE(ones_vect)

        if combine:
            mask = tf.reduce_mean(mask, axis=0)
            choose = np.where(choose.numpy() > threshold, 1, 0)
            #print('RATIO: ', np.mean(choose, axis=0))
            choose = np.where(np.mean(choose, axis=0)>acceptance_ratio, 1, 0)
            patches = self.MASKAPPLY([sample[0][0:1], mask, choose])
        else:
            #print(choose.numpy()[:2])
            choose = np.where(choose.numpy() > threshold, 1, 0)
            patches = self.MASKAPPLY([sample[0], mask, choose])

        return patches.numpy(), choose
