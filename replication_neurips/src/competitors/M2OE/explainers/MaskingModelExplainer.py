from math import ceil

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np


class MaskingModelExplainer(keras.Model):
    """
    Abstract class
    """

    def __init__(self, loss_weights):
        super(MaskingModelExplainer, self).__init__()
        self.MASK = None
        self.CHOOSE = None
        self.MASKAPPLY = None
        self.PATCH = None
        self.loss_weights = loss_weights

    def defineMaskGen(self, in_shape):
        """
        Define the mask generator model, returns NotImplementedError if it is called from abstract class
        :param in_shape: input shape
        :return:
        """
        raise NotImplementedError('subclasses must override defineMaskGen!')

    def defineMaskApply(self, in_shape):
        """
        Define the mask applier model, returns NotImplementedError if it is called from abstract class
        :param in_shape: input shape
        :return:
        """

        raise NotImplementedError('subclasses must override defineMaskApply!')

    def definePatch(self, in_shape):
        """
        Define the model that produce the patch from the original image
        :param in_shape: input shape
        :return:
        """

        # Mette insieme maskapply e maskgen
        img_o = keras.Input(shape=in_shape)
        img_i = keras.Input(shape=in_shape)

        self.defineMaskGen(in_shape)
        self.defineMaskApply(in_shape)

        mask = self.MASK([img_o, img_i])
        choose = self.CHOOSE([img_o, img_i])
        patch = self.MASKAPPLY([img_o, mask, choose])

        # Generatore patch - Applicatore della patch - Modello completo (gen+app)
        self.PATCH = keras.Model(inputs=[img_o, img_i], outputs=[patch, mask, choose])
        return

    def test(self, id, classes, train_images, train_labels, drawplot=True):

        raise NotImplementedError('subclasses must override test!')

    def buildExplanator(self, in_shape):
        """
        Build the explanator model
        :param in_shape: input shape
        :return:
        """
        self.definePatch(in_shape)
        return

    def explain(self, sample, threshold=0.7, acceptance_ratio=0.5):
        raise NotImplementedError('subclass must override explain!')

