# data loader script for training unets
from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils.data_utils import Sequence
import h5py
from tensorflow.python.keras.utils.data_utils import Sequence
import h5py

class dataLoader3D(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 x_path, 
                 y_path, 
                 data_type='train', 
                 batch_size=48, 
                 start=1, stop=90,
                 shuffle=False, 
                 nu_indx=None
                    ):
        
        
        'Initialization'
        self.data_type = data_type + '_data'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.y_fname = y_path + 'cosmo.h5'    # y filename doesn't change

        self.x_path = x_path
        self.x_fname = self.x_path 


    def __len__(self):
        return int(np.floor(((self.stop-self.start)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        x_fname = self.x_path +  'pca3_%d.h5'%(int(np.ceil(np.random.rand()*5)))
 
        batch_x = self.load_data(x_fname, idx)
        batch_y = self.load_data(self.y_fname, idx)
        
        return batch_x, batch_y
    
    def __gettruth__(self):
        return h5py.File(self.y_fname, 'r')[self.data_type][self.start*self.nwinds:self.stop*self.nwinds]

    
    def load_data(self, fname, idx):
        d = h5py.File(fname, 'r')[self.data_type]
        d = d[idx * self.batch_size:(idx + 1) * self.batch_size]
        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T        
        d = np.expand_dims(d, axis=-1)
        return d

class dataLoader3D_static(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_path, y_path, batch_size=48, 
                    start=1, stop=20,
                    shuffle=False, nu_indx=None):
        
        
        'Initialization'
        self.start = start + 1
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.y_fnames = [y_path + 'cosmo/cosmo_nnu032_sim%03d.npy'%(i) for i in range(start+1, stop)]
        self.x_fnames = [x_path + 'pca3_%d/pca_3comp_nnu32_sim%03d.npy'%(int(np.ceil(np.random.rand()*5)),i) for i in range(start+1, stop)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.x_data = np.concatenate([self.load_data(p) for p in self.x_fnames], axis=0)
        self.y_data = np.concatenate([self.load_data(p) for p in self.y_fnames], axis=0)


    def __len__(self):
        return int(np.floor(len(self.x_data) // self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
    
    def __gettruth__(self):
        return np.squeeze(self.y_data)

    
    def load_data(self, fname):
        d = np.load(fname)
        # shuffle frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T
        
        d = np.expand_dims(d, axis=-1)
        return d


class dataLoader2D(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 x_path, 
                 y_path, 
                 data_type='train', 
                 batch_size=48, 
                 start=1, stop=90,
                 shuffle=False, 
                 nu_indx=None
                    ):
        
        
        'Initialization'
        self.data_type = data_type + '_data'
        self.start = start
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.y_fname = y_path + 'cosmo.h5'    # y filename doesn't change

        self.x_path = x_path
        self.x_fname = self.x_path 


    def __len__(self):
        return int(np.floor(((self.stop-self.start)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        x_fname = self.x_path +  'pca3_%d.h5'%(int(np.ceil(np.random.rand()*5)))
 
        batch_x = self.load_data(x_fname, idx)
        batch_y = self.load_data(self.y_fname, idx)
        
        return batch_x, batch_y
    
    def __gettruth__(self):
        return h5py.File(self.y_fname, 'r')[self.data_type][self.start*self.nwinds:self.stop*self.nwinds]

    
    def load_data(self, fname, idx):
        d = h5py.File(fname, 'r')[self.data_type]
        d = d[idx * self.batch_size:(idx + 1) * self.batch_size]
        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T        
        #d = np.expand_dims(d, axis=-1)
        return d



class dataLoader2D_static(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, x_path, y_path, batch_size=48, 
                    start=1, stop=20,
                    shuffle=False, nu_indx=None):
        
        
        'Initialization'
        self.start = start + 1
        self.stop = stop
        self.nwinds = 768          # simulation param, num bricks per sim
        self.y_fnames = [y_path + 'cosmo/cosmo_nnu032_sim%03d.npy'%(i) for i in range(start+1, stop)]
        self.x_fnames = [x_path + 'pca3_%d/pca_3comp_nnu32_sim%03d.npy'%(int(np.ceil(np.random.rand()*5)),i) for i in range(start+1, stop)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nu_indx = nu_indx
        self.x_data = np.concatenate([self.load_data(p) for p in self.x_fnames], axis=0)
        self.y_data = np.concatenate([self.load_data(p) for p in self.y_fnames], axis=0)


    def __len__(self):
        return int(np.floor(len(self.x_data) // self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y_data[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y
    
    def __gettruth__(self):
        return np.squeeze(self.y_data)

    
    def load_data(self, fname):
        d = np.load(fname)
        # shuffle frequencies if desired with input indexes
        if self.nu_indx is not None:
            d = d.T[self.nu_indx].T
        
        return d