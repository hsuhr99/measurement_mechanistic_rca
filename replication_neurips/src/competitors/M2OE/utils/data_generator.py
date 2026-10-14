import numpy as np
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, batch_size, shuffle=True):
        'Initialization'
        self.labels = labels
        self.data = data[labels==0]
        self.shuffle = shuffle
        self.anomalous = data[labels==1]
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.data[indexes]
        y = self.labels[indexes]

        # Insert anomalous data
        X = np.append(X, self.anomalous, axis=0)
        y = np.append(y, np.ones(self.anomalous.shape[0]), axis=0)

        return X, y