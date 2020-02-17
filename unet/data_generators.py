"""
Script for organizing Keras classes, such per-epoch x_data generators

T Lucas Makinen
"""
import numpy as np
from tensorflow import keras

# data grouped in tensors of shape (64,64,64,3), corresponding to (x,y,nu,3)
# data obtained from simulation for which 1 sky = 192 input cubes of (64,64,64)
# => sky_batch_size = 192 / 3 = 96

class tomo_dataLoader(keras.utils.Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, path, batch_size=64, x_dim=(64,64,64,3), y_dim=(64, 64,64,3), n_channels=1, numskies=2,
                         n_skip=2, shuffle=False):
        'Initialization'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.numskies = numskies
        self.n_skip = n_skip       # for cutting down input resolution
        self.nwinds = int(192 / 3)         # simulation param
        self.path = path
        self.y_files = [path + 'cosmo_%03d'%(i) + '.npy' for i in range(numskies*self.nwinds)]
        self.x_files = [path + 'pca_%03d'%(i) + '.npy' for i in range(numskies*self.nwinds)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'        
        return int(np.floor(len(self.x_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of x_data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_indexes_temp = [self.x_files[k] for k in indexes]
        y_indexes_temp = [self.y_files[k] for k in indexes]
        
        #print(x_indexes_temp)

        # Generate data
        X, y = self.__x_data_generation(x_indexes_temp, y_indexes_temp)
        #sample = {'x_array': get_minibatch(self.x_files[index]), 'y_array': get_minibatch(self.y_files[index])}

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x_files))
        'Updates indexes after each epoch in units of sky'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __x_data_generation(self, x_indexes_temp, y_indexes_temp):
        'Generates x_data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, *self.y_dim))

        # Generate x_data
        for i in range(len(x_indexes_temp)):
            # Store x data
            X[i,] = self.get_minibatch(x_indexes_temp[i])
            y[i,] = self.get_minibatch(y_indexes_temp[i])

        return X, y
    
    def get_minibatch(self, fname):
        data = np.load(fname)
        # can add data augmentation here

        data = data[::self.n_skip,::self.n_skip,::self.n_skip,:]
        return data



class EpochDataGenerator(keras.utils.Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_data, y_data, batch_size=192, x_dim=(64,64,30), y_dim=(64, 64, 30), n_channels=1, num_sims=20,
                         num_train=78, shuffle=True):
        'Initialization'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.num_sims = num_sims  # number of sky simlations chosen at random per epoch
        self.num_train = num_train
        self.y_data = y_data  
        self.x_data = x_data
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return 1 # one random 1000-sim batch per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of x_data'
        # Generate indexes of the batch
        indx = self.rand[index]            # choose value from rand_indexes
        #_range = np.arange(len(self.x_data))       # create range array of length of x-data
        indexes = self.indexes[(indx-1)*self.batch_size:(indx)*self.batch_size]  # start from indx and select an ordered, batch-size sample
        # Find list of IDs
        x_data_temp = [self.x_data[k] for k in indexes]
        y_data_temp = [self.y_data[k] for k in indexes]

        # Generate x_data
        X, y = self.__x_data_generation(x_data_temp, y_data_temp)
        
        #print('y input shapes: ', np.array(y_data_temp).shape)
        #print(indexes)
        #print('X max : ', np.max(X))
        #print('y shape : ', y.shape)

        return X, y

    def on_epoch_end(self):

        'Updates indexes after each epoch in units of sky'
        self.indexes = np.arange(self.batch_size * self.num_train)
        # select num_sims subset of full dataset each epoch
        self.rand = np.random.choice(np.arange(self.num_train), size=self.num_sims)
        #np.arange(len(self.x_data))
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def __x_data_generation(self, x_data_temp, y_data_temp):
        'Generates x_data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        y = np.empty((self.batch_size, *self.y_dim))

        # Generate x_data
        for i in range(len(x_data_temp)):
            # Store x data
            X[i,] = x_data_temp[i] 
            y[i,] = y_data_temp[i]

        return X, y




class EpochDataGenNetwork2(keras.utils.Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_data, y_data, batch_size=64, x_dim=(64,64,30), y_dim=(64, 64, 30), n_channels=1, num_sims=30,
                         num_train=78, shuffle=True):
        'Initialization'
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.num_sims = num_sims  # number of simlations chosen at random per epoch
        self.num_train = num_train
        self.fg_data = y_data[0]  
        self.cosmo_data = y_data[1]
        self.x_data = x_data
        self.n_channels = n_channels
        #self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return 1 # one random 1000-sim batch per epoch
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of x_data'
        # Generate indexes of the batch
        indx = self.rand_indexes[index]            # choose value from rand_indexes
        _range = np.arange(len(self.x_data))       # create range array of length of x-data
        indexes = _range[indx*self.batch_size:(indx+1)*self.batch_size]  # start from indx and select an ordered, batch-size sample
        # Find list of IDs
        x_data_temp = [self.x_data[k] for k in indexes]
        fg_data_temp = [self.fg_data[k] for k in indexes]
        cosmo_data_temp = [self.cosmo_data[k] for k in indexes]

        # Generate x_data
        X, fg_data, cosmo_data = self.__x_data_generation(x_data_temp, fg_data_temp, cosmo_data_temp)


        y = [fg_data, cosmo_data]
        
        print('x', x_data_temp)
        #print(indexes)
        #print('X max : ', np.max(X))
        #print('y shape : ', y.shape)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch in units of sky'
        indexes = np.arange(self.num_train)
        # select num_sims subset of full dataset each epoch
        self.rand_indexes = np.random.choice(indexes, size=self.num_sims)
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def __x_data_generation(self, x_data_temp, fg_data_temp, cosmo_data_temp):
        'Generates x_data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.x_dim))
        fg_data = np.empty((self.batch_size, *self.y_dim))
        cosmo_data = np.empty((self.batch_size, *self.y_dim))

        # Generate x_data
        for i in range(len(x_data_temp)):
            # Store x data
            X[i,] = x_data_temp[i] 
            cosmo_data[i,] = cosmo_data_temp[i]
            fg_data[i,] = fg_data_temp[i]

        return X, fg_data, cosmo_data